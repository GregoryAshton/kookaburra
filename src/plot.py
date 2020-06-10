import bilby
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_rcparams():
    # Setup some matplotlib defaults
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "Computer Modern"
    mpl.rcParams["text.usetex"] = "True"
    mpl.rcParams["text.latex.preamble"] = r"\newcommand{\mathdefault}[1][]{}"


def plot_data(data, ax=None, reference_time=None, label=None, filename=None,
              time_prior=None):
    """ Plot the data

    Parameters
    ----------
    data: kookaburra.data.TimeDomainData
        The data to plot

    """

    if reference_time is not None:
        data.reference_time = reference_time

    if ax is None:
        fig, ax = plt.subplots()

    # Plot the data
    ax.plot(data.delta_time, data.flux, label=label, lw=1, color="C0", zorder=-100)

    ax.set_ylabel("Flux")
    if data.reference_time != 0:
        ax.set_xlabel(f"Time - {data.reference_time} [{data.time_unit}]")
    else:
        ax.set_xlabel(f"Time [{data.time_unit}]")

    if time_prior is not None:
        ax.axvspan(time_prior.minimum - data.reference_time,
                   time_prior.maximum - data.reference_time,
                   alpha=0.1, color='k')

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
    else:
        return ax


def plot_fit(data, result, model, priors, outdir, label, width="auto"):
    """ Plot the data and the fit and a residual

    Parameters
    ----------
    result: bilby.core.result.Result, None
        The result object to show alongside the data. If given as None,
        only the raw data is plotted.
    model: function
        Function fitted to the data
    priors: dict
        Dictionary of the priors used
    outdir, label: str
        An outdir and label to use in generating the filename
    tref: float, None
        A reference time to subtract of to improve visualization. If a
        result is given, then the median TOA is used instead. If None is
        given, then the mid-point is used.

    """

    # Max likelihood sample
    maxl_sample = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]

    # Set the reference time to the max-l TOA
    data.reference_time = maxl_sample["toa"]
    times = data.delta_time

    # Calculate the max likelihood flux
    maxl_flux = model(data.time, **maxl_sample)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, figsize=(5, 4),
        gridspec_kw=dict(height_ratios=[2, 1]))

    # Plot the time prior window
    for ax in [ax1, ax2]:
        ax.axvspan(
            priors["toa"].minimum - data.reference_time,
            priors["toa"].maximum - data.reference_time,
            color='k', alpha=0.05)

    # Plot the data
    ax1.plot(times, data.flux, label="data", lw=1, color="C0", zorder=-100)

    # Plot the maximum likelihood
    ax1.plot(times, maxl_flux, lw=0.5, color="C2", zorder=100)

    # Plot the 90%
    npreds = 100
    nsamples = len(result.posterior)
    preds = np.zeros((npreds, len(times)))
    for ii in range(npreds):
        draw = result.posterior.iloc[np.random.randint(nsamples)]
        preds[ii] = model(data.time, **draw)
    ax1.fill_between(
        times,
        np.quantile(preds, q=0.05, axis=0),
        np.quantile(preds, q=0.95, axis=0),
        color="C1", alpha=0.8, zorder=0)

    # Plot the sigma uncertainty on the residual
    median_sigma = np.median(result.posterior["sigma"])
    ax2.axhspan(-median_sigma, median_sigma, color='k', alpha=0.2)

    # Plot the 90% residual
    res_preds = data.flux - preds
    ax2.fill_between(
        times,
        np.quantile(res_preds, 0.05, axis=0),
        np.quantile(res_preds, 0.95, axis=0),
        color='C1', alpha=0.5, zorder=0)
    ax2.plot(times, data.flux - maxl_flux, "C0", lw=0.5, zorder=100)

    if width == "auto":
        # Auto-zoom to interesting region
        maxl_pulse = model.get_pulse_only(data.time, **maxl_sample)
        maxl_peak = np.max(maxl_pulse)
        times_near_peak = times[maxl_pulse > 1e-3 * maxl_peak]
        if len(times_near_peak) > 1:
            ax1.set_xlim(
                np.min(times_near_peak), np.max(times_near_peak))
    elif width == "full":
        pass
    else:
        width = np.float(width)
        ax1.set_xlim(- .5 * width, + .5 * width)

    ax1.set_ylabel("Flux")
    ax2.set_xlabel(f"Time - {data.reference_time} [{data.time_unit}]")
    ax2.set_ylabel("Flux residual")

    filename = "{}/{}_fit_with_data.png".format(outdir, label)

    fig.tight_layout()
    fig.savefig(filename, dpi=600)


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


def plot_result_null_corner(result, result_null, args):
    parameters = [key for key in result_null.priors]
    bilby.core.result.plot_multiple(
        [result, result_null],
        parameters=parameters,
        labels=[f"{args.label}", f"{args.label}_null"],
        filename=f"{args.outdir}/{args.label}_baseflux_corner",
        priors=True)



