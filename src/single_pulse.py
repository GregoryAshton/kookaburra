""" Command line tool for single-pulse shapelet analysis """
import argparse
import logging

import bilby
from scipy.stats import normaltest

from . import flux
from . import plot
from .priors import update_toa_prior
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
        "--truncate-data", type=float, default=None)

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
        nargs="+",
        required=True,
        help=("Required: the number of shapelets to fit. Multiple component "
              "models are specified by a list, e.g. `-s 2 3 1`")
    )
    parser.add_argument(
        "-b", "--base-flux-n-polynomial", default=1, type=int,
        help="The order for the base polynomial"
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
    plot_parser.add_argument("--pretty", action="store_true", help="Use latex for plotting")
    plot_parser.add_argument("--max-corner", default=6, help="Maximum number of components in corner plots")

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
        "--toa-prior-width", type=float, default=1,
        help="Duration fraction for time prior. If 1, the whole data span used."
    )
    prior_parser.add_argument(
        "--toa-prior-time", type=str, default="auto",
        help=("If a float [0, 1], the fraction of the duration to place the "
              "centre of the time window. If auto (default) the time is taken "
              "from the peak. If toa-prior-width=1, this argument is redudant")
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

    priors = full_model.get_priors(data)
    priors = add_sigma_prior(priors, data)
    priors = update_toa_prior(priors)

    # Pre-plot the data
    if args.plot_data:
        plot.plot_data(data, filename=f"{args.outdir}/{args.label}_data",
                       time_priors=[p for k, p in priors.items() if "toa" in k])

    likelihood = PulsarLikelihood(
        data, full_model, noise_log_likelihood=result_null.log_evidence)

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
    result.meta_data["maxl_residual"] = residual
    result.meta_data["maxl_normaltest_pvalue"] = normaltest(residual).pvalue

    if args.plot_corner:
        for model in full_model.models:
            parameters = model.parameters
            if len(parameters) == 0:
                continue
            if len(parameters) <= args.max_corner:
                filename = f"{args.outdir}/{args.label}_{model.name}_corner"
                result.plot_corner(
                    parameters=parameters, priors=True, filename=filename
                )

            plot.plot_coeffs(result, args, model)
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

    for ii, ns in enumerate(args.n_shapelets):
        ns = int(ns)
        if ns > 0:
            name = f"S{ii}"
            args.label += f"_{name}-{ns}"
            full_model += flux.ShapeletFlux(
                n_shapelets=ns, toa_prior_width=args.toa_prior_width,
                toa_prior_time=args.toa_prior_time, c_mix=args.c_mix,
                c_max_multiplier=args.c_max_multiplier, beta_type=args.beta_type,
                beta_min=args.beta_min, beta_max=args.beta_max,
                name=name)

    if args.base_flux_n_polynomial > 0:
        args.label += f"_BP{args.base_flux_n_polynomial}"
        null_model += flux.PolynomialFlux(args.base_flux_n_polynomial, name="BP")
        full_model += flux.PolynomialFlux(args.base_flux_n_polynomial, name="BP")

    logger.info(f"Reading data for pulse {args.pulse_number} from {args.data_file}")
    data = TimeDomainData.from_file(
        args.data_file, pulse_number=args.pulse_number)

    if args.truncate_data is not None:
        data.truncate_data(args.truncate_data)

    result_null = run_null_analysis(args, data, null_model)
    run_full_analysis(args, data, full_model, result_null)
