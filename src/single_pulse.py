""" Command line tool for single-pulse shapelet analysis """
import argparse
import os
import logging

import bilby
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .flux_model import SinglePulseFluxModel
from .data import TimeDomainData
from .likelihood import PulsarLikelihood, NullLikelihood
from .priors import get_priors

# Setup some matplotlib defaults
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Computer Modern"
mpl.rcParams["text.usetex"] = "True"
mpl.rcParams["text.latex.preamble"] = r"\newcommand{\mathdefault}[1][]{}"


def get_args():
    parser = argparse.ArgumentParser(
        description="Run single pulse analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data_file", type=str, help="The data file")

    shape_parser = parser.add_argument_group("Shapelets options")
    shape_parser.add_argument(
        "-p",
        "--pulse-number",
        type=int,
        default=None,
        required=False,
        help="The pulse number to analyse. If not given, no pulse-number filter is applied",
    )
    shape_parser.add_argument(
        "-s",
        "--n-shapelets",
        type=int,
        required=True,
        help="Required: the number of shapelets to fit.",
    )

    plot_parser = parser.add_argument_group("Output options")
    plot_parser.add_argument(
        "--plot-corner", action="store_true", help="Create corner plots"
    )
    plot_parser.add_argument(
        "--plot-fit", action="store_true", help="Create residual plots"
    )
    plot_parser.add_argument("--plot-run", action="store_true", help="Create run plots")

    prior_parser = parser.add_argument_group("Prior options")
    prior_parser.add_argument(
        "--no-base-flux", action="store_true", help="Fix base flux to zero"
    )
    prior_parser.add_argument(
        "--beta-min", type=float, default=1e-10, help="Minimum beta value"
    )
    prior_parser.add_argument(
        "--beta-max", type=float, default=1e-6, help="Maximum beta value"
    )
    prior_parser.add_argument(
        "--beta-type", type=str, default="uniform", help="Beta-prior",
        choices=["uniform", "log-uniform"]
    )
    prior_parser.add_argument(
        "--c-max-multiplier",
        type=float,
        default=1,
        help="Multiplier of the max flux to use for setting the coefficient upper bound",
    )
    prior_parser.add_argument(
        "--c-mix", type=float, default=0.5, help="Mixture between spike and slab"
    )
    prior_parser.add_argument(
        "--toa-width", type=float, default=1,
        help="Duration fraction for time prior. If 1, the whole data span used."
    )

    sampler_parser = parser.add_argument_group("Sampler options")
    sampler_parser.add_argument(
        "--sampler", type=str, default="pymultinest", help="Sampler to use"
    )
    sampler_parser.add_argument(
        "--nlive", type=int, default=1000, help="Number of live points to use"
    )
    sampler_parser.add_argument(
        "--sampler-kwargs", type=str,
        help="Arbitrary kwargs dict to pass to the sampler"
    )


    args, _ = parser.parse_known_args()

    return args


def run_analysis(args, data, model, priors):
    likelihood = PulsarLikelihood(data, model)

    run_sampler_kwargs = dict(
        sampler=args.sampler, nlive=args.nlive)

    if args.sampler_kwargs:
        run_sampler_kwargs.update(eval(args.sampler_kwargs))


    result = bilby.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        label=args.label,
        save=False,
        outdir=args.outdir,
        check_point_plot=args.plot_run,
        **run_sampler_kwargs
    )

    s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
    residual = data.flux - model(data.time, **s)

    priors_null = bilby.core.prior.PriorDict()
    priors_null["sigma"] = priors["sigma"]
    priors_null["base_flux"] = priors["base_flux"]
    likelihood_null = NullLikelihood(data)
    result_null = bilby.sampler.run_sampler(
        likelihood=likelihood_null,
        priors=priors_null,
        label=args.label + "_null",
        outdir=args.outdir,
        save=False,
        check_point=True,
        check_point_plot=False,
        verbose=False,
        **run_sampler_kwargs
    )

    result.log_noise_evidence = result_null.log_evidence
    result.log_noise_evidence_err = result_null.log_evidence_err
    result.meta_data["args"] = args.__dict__
    result.meta_data["residual"] = residual
    result.meta_data["RMS_residual"] = np.sqrt(np.mean(residual ** 2))
    result.save_to_file()

    return result, result_null


def save(args, data, result, result_null, outdir):
    rows = [
        "pulse_number",
        "toa",
        "toa_std",
        "base_flux",
        "base_flux_std",
        "beta",
        "beta_std",
        "sigma",
        "sigma_std",
        "log_evidence",
        "log_evidence_err",
        "log_noise_evidence",
        "log_noise_evidence_err",
        "toa_prior_width",
        "normal_p_value",
    ]
    for i in range(args.n_shapelets):
        rows.append("C{}".format(i))
        rows.append("C{}_err".format(i))

    filename = f"{outdir}/{args.n_shapelets}_shapelets.summary"
    if os.path.isfile(filename) is False:
        with open(filename, "w+") as f:
            f.write(",".join(rows) + "\n")

    p = result.posterior
    toa_prior_width = result.priors["toa"].maximum - result.priors["toa"].minimum
    row_list = [
        args.pulse_number,
        p.toa.median(),
        p.toa.std(),
        p.base_flux.median(),
        p.base_flux.std(),
        p.beta.median(),
        p.beta.std(),
        p.sigma.median(),
        p.sigma.std(),
        result.log_evidence,
        result.log_evidence_err,
        result_null.log_evidence,
        result_null.log_evidence_err,
        toa_prior_width,
        data.normal_pvalue,
    ]
    for i in range(args.n_shapelets):
        row_list.append(p["C{}".format(i)].mean())
        row_list.append(p["C{}".format(i)].std())

    with open(filename, "a") as f:
        f.write(",".join([str(el) for el in row_list]) + "\n")


def plot_coeffs(result, args):
    coeffs = [f"C{ii}" for ii in range(1, args.n_shapelets)]
    samples = result.posterior[coeffs].values
    bins = np.linspace(np.min(samples[samples > 0]), np.max(samples))
    fig, ax = plt.subplots()
    for CC in coeffs:
        ax.hist(result.posterior[CC], bins=bins, alpha=0.5, label=CC)
    ax.set_xlabel("Coefficient amplitudes")
    ax.legend()
    fig.savefig(f"{args.outdir}/{args.label}_coefficients")


def main():

    logger = logging.getLogger('single_pulse')
    logger.setLevel(logging.INFO)

    args = get_args()

    args.outdir = "outdir_single_pulse"
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)
    args.label = "pulse_{}_shapelets_{}".format(args.pulse_number, args.n_shapelets)

    model = SinglePulseFluxModel(n_shapelets=args.n_shapelets)

    logger.info(f"Reading data from {args.data_file}")
    data = TimeDomainData.from_file(args.data_file, pulse_number=args.pulse_number)

    priors = get_priors(args, data)

    # Pre-plot the data and prior window
    if args.plot_fit:
        logger.info("Pre-plot the data")
        data.plot_fit(None, model, priors, outdir=args.outdir, label=args.label)

    logger.info("Run the analysis")
    result, result_null = run_analysis(args, data, model, priors)

    if args.plot_corner:
        parameters = ["toa", "beta", "sigma", "C0"]
        result.plot_corner(parameters=parameters, priors=True)
        if args.n_shapelets > 1:
            plot_coeffs(result, args)

    if args.plot_fit:
        data.plot_fit(result, model, priors, outdir=args.outdir, label=args.label)

    save(args, data, result, result_null, args.outdir)
