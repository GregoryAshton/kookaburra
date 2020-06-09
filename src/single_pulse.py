""" Command line tool for single-pulse shapelet analysis """
import argparse
import os
import logging

import bilby
import numpy as np

from . import flux
from . import plot
from .data import TimeDomainData
from .likelihood import PulsarLikelihood


logger = logging.getLogger('single_pulse')
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run single pulse analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data_file", type=str, help="The data file")

    parser.add_argument("--outdir", type=str, help="The output directory",
                        default="outdir")
    parser.add_argument("--label", type=str, help="Extra label elements",
                        default=None)

    parser.add_argument(
        "-p",
        "--pulse-number",
        type=int,
        default=None,
        required=False,
        help=("The pulse number to analyse. If not given, no pulse-number "
              "filter is applied"),
    )
    parser.add_argument(
        "-s",
        "--n-shapelets",
        type=int,
        required=True,
        help="Required: the number of shapelets to fit.",
    )
    parser.add_argument(
        "-b", "--base-flux-n-polynomial", default=1, type=int,
    )

    plot_parser = parser.add_argument_group("Output options")
    plot_parser.add_argument(
        "--plot-corner", action="store_true", help="Create corner plots"
    )
    plot_parser.add_argument(
        "--plot-fit", action="store_true", help="Create residual plots"
    )
    plot_parser.add_argument(
        "--plot-data", action="store_true", help="Create initial data plots"
    )
    plot_parser.add_argument(
        "--plot-fit-width", type=str, default='auto',
        help="Width of the fit plot. Options: `auto` or a float (fixed width)"
    )
    plot_parser.add_argument(
        "--plot-run", action="store_true",
        help="Create run plots if available")
    plot_parser.add_argument("--pretty", action="store_true", help="")

    prior_parser = parser.add_argument_group("Prior options")
    prior_parser.add_argument(
        "--beta-min", type=float, default=None, help="Minimum beta value"
    )
    prior_parser.add_argument(
        "--beta-max", type=float, default=None, help="Maximum beta value"
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
        "--c-mix", type=float, default=0.1, help="Mixture between spike and slab"
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
    sampler_parser.add_argument(
        "-c", "--clean", action="store_true",
    )

    args = parser.parse_args()

    return args


def add_sigma_prior(priors, data):
    priors['sigma'] = bilby.core.prior.Uniform(
        0, data.range_flux, 'sigma', latex_label=r"$\sigma$")
    return priors


def get_sampler_kwargs(args):
    run_sampler_kwargs = dict(
        sampler=args.sampler, nlive=args.nlive)

    if args.sampler_kwargs:
        run_sampler_kwargs.update(eval(args.sampler_kwargs))

    return run_sampler_kwargs


def run_full_analysis(args, data, full_model, result_null):

    # Pre-plot the data and prior window
    if args.plot_data:
        plot.plot_data(data, filename=f"{args.outdir}/{args.label}_data")

    likelihood = PulsarLikelihood(
        data, full_model, noise_log_likelihood=result_null.log_evidence)

    priors = full_model.get_priors(data)
    priors = add_sigma_prior(priors, data)

    result = bilby.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        label=args.label,
        save=False,
        outdir=args.outdir,
        check_point_plot=args.plot_run,
        clean=args.clean,
        **get_sampler_kwargs(args)
    )

    s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
    residual = data.flux - full_model(data.time, **s)
    result.meta_data["args"] = args.__dict__
    result.meta_data["residual"] = residual
    result.meta_data["RMS_residual"] = np.sqrt(np.mean(residual ** 2))

    if args.plot_corner:
        parameters = ["toa", "beta", "sigma", "C0"]
        result.plot_corner(parameters=parameters, priors=True)
        if args.n_shapelets > 1:
            plot.plot_coeffs(result, args)
        plot.plot_result_null_corner(result, result_null, args)

    if args.plot_fit:
        plot.plot_fit(
            data, result, full_model, priors, outdir=args.outdir,
            label=args.label, width=args.plot_fit_width)

    result.log_noise_evidence_err = result_null.log_evidence_err
    result.save_to_file()

    return result


def run_null_analysis(args, data, null_model):
    priors = null_model.get_priors(data)
    priors = add_sigma_prior(priors, data)

    likelihood_null = PulsarLikelihood(data, null_model)
    result = bilby.sampler.run_sampler(
        likelihood=likelihood_null,
        priors=priors,
        label=args.label + "_null",
        outdir=args.outdir,
        save=False,
        check_point=True,
        check_point_plot=False,
        clean=args.clean,
        verbose=False,
        **get_sampler_kwargs(args)
    )

    return result


def save(args, data, result, result_null, outdir):
    rows = [
        "pulse_number",
        "toa",
        "toa_std",
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

    for i in range(args.base_flux_n_polynomial):
        rows.append("B{}".format(i))
        rows.append("B{}_err".format(i))

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

    for i in range(args.base_flux_n_polynomial):
        row_list.append(p["B{}".format(i)].mean())
        row_list.append(p["B{}".format(i)].std())

    with open(filename, "a") as f:
        f.write(",".join([str(el) for el in row_list]) + "\n")


def main():
    args = get_args()

    if args.pretty:
        plot.set_rcparams()

    if args.label is None:
        args.label = f"pulse_{args.pulse_number}"
    else:
        args.label = f"{args.label}_pulse_{args.pulse_number}"

    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    full_model = flux.BaseFlux()
    null_model = flux.BaseFlux()

    if args.n_shapelets > 0:
        args.label += f"_S{args.n_shapelets}"
        full_model += flux.ShapeleteFlux(
            n_shapelets=args.n_shapelets, toa_width=args.toa_width,
            c_mix=args.c_mix, c_max_multiplier=args.c_max_multiplier,
            beta_type=args.beta_type, beta_min=args.beta_min,
            beta_max=args.beta_max)

    if args.base_flux_n_polynomial > 0:
        args.label += f"_P{args.n_shapelets}"
        null_model += flux.PolynomialFlux(args.base_flux_n_polynomial)
        full_model += flux.PolynomialFlux(args.base_flux_n_polynomial)

    logger.info(f"Reading data for pulse {args.pulse_number} from {args.data_file}")
    data = TimeDomainData.from_file(
        args.data_file, pulse_number=args.pulse_number)

    result_null = run_null_analysis(args, data, null_model)
    result_full = run_full_analysis(args, data, full_model, result_null)
    save(args, data, result_full, result_null, args.outdir)


