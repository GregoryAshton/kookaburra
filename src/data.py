import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest


class TimeDomainData:
    """ Object to store time-domain flux data """

    def __init__(self):
        pass

    @property
    def start(self):
        """ Data start time """
        return self.time[0]

    @property
    def end(self):
        """ Data end time """
        return self.time[-1]

    @property
    def duration(self):
        """ Data duration """
        return self.end - self.start

    @property
    def N(self):
        """ Number of data points """
        return len(self.time)

    @property
    def RMS_flux(self):
        """ Return the root-mean-square of all flux """
        return np.sqrt(np.mean(self.flux ** 2))

    @property
    def max_flux(self):
        """ Return the maximum flux """
        return np.max(self.flux)

    @property
    def max_time(self):
        """ Return the time of the maximum flux """
        return self.time[np.argmax(self.flux)]

    def estimate_pulse_time(self, f=0.75):
        """ Estimate the pulse time """
        idxs = np.abs(self.flux) > f * self.max_flux
        return np.mean(self.time[idxs])

    def estimate_pulse_width(self, f=0.75):
        """ Estimate the pulse time """
        idxs = np.abs(self.flux) > f * self.max_flux
        return np.std(self.time[idxs])

    @classmethod
    def from_array(cls, time, flux):
        """ Read in the time and flux from a csv

        Parameters
        ----------
        time, flux: np.ndarray
            The time and flux arrays

        """

        if time.shape != flux.shape:
            raise ValueError("TimeDomainData only valid equal-shape arrays")
        if time.ndim > 1:
            raise ValueError("TimeDomainData only valid for single-dimensional data")
        if np.any(np.diff(time) < 0):
            raise ValueError("TimeDomainData requires sorted data")
        time_domain_data = TimeDomainData()
        time_domain_data.time = time
        time_domain_data.flux = flux
        return time_domain_data

    @classmethod
    def from_csv(cls, filename, pulse_number=None):
        """ Read in the time and flux from a csv

        Parameters
        ----------
        filename: str
            The path to the file to read
        pulse_number: int:
            The pulse number to truncate.

        """
        df = pd.read_csv(filename)
        return cls._sort_and_filter_dataframe(df, pulse_number)

    @classmethod
    def from_file(cls, filename, pulse_number=None):
        """ Read in the time and flux from file

        Parameters
        ----------
        filename: str
            The path to the file to read
        pulse_number: int:
            The pulse number to truncate.

        """
        if "h5" in filename:
            return cls.from_h5(filename, pulse_number)
        else:
            return cls.from_csv(filename, pulse_number)

    @classmethod
    def from_h5(cls, filename, pulse_number=None):
        """ Read in the time and flux from a pandas h5 file

        Parameters
        ----------
        filename: str
            The path to the file to read
        pulse_number: int:
            The pulse number to truncate.

        """
        df = pd.read_hdf(filename)
        return cls._sort_and_filter_dataframe(df, pulse_number)

    @staticmethod
    def _sort_and_filter_dataframe(df, pulse_number):
        df = df.sort_values("time")
        if pulse_number is not None:
            df = df[df.pulse_number == pulse_number]
        time_domain_data = TimeDomainData()
        time_domain_data.time = df.time.values
        time_domain_data.flux = df.flux.values
        del df
        return time_domain_data

    def plot_max_likelihood(self, result=None, model=None, xlims=None):
        """ Plot the data and max-likelihood

        Parameters
        ----------
        result: bilby.core.result.Result
            The result object to show alongside the data
        model: function
            Function fitted to the data

        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(self.time, self.flux, label="data", lw=2)

        # Plot the maximum likelihood
        s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
        ax1.plot(self.time, model(self.time, **s), "--", label="max-l model")
        ax1.legend()
        ax1.set_ylabel("Flux")

        ax2.plot(self.time, self.flux - model(self.time, **s))
        ax2.set_xlabel("time")
        ax2.set_ylabel("Flux residual")

        if xlims is not None:
            ax1.set_xlim(*xlims)

        fig.tight_layout()
        fig.savefig(
            "{}/{}_maxl_with_data.png".format(result.outdir, result.label), dpi=500
        )

    def plot_fit(self, result, model, priors, outdir, label, tref=None):
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

        if result:
            s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
            maxl = model(self.time, **s)
            tref = s["toa"]  # Set the reference time to the max-l TOA
        else:
            tref = .5 * (self.start + self.end)

        times = self.time - tref

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, figsize=(5, 4),
            gridspec_kw=dict(height_ratios=[2, 1]))

        # Plot the time prior window
        for ax in [ax1, ax2]:
            ax.axvspan(
                priors["toa"].minimum - tref,
                priors["toa"].maximum - tref,
                color='k', alpha=0.1)

        # Plot the data
        ax1.plot(times, self.flux, label="data", lw=1, color="C0", zorder=-100)

        if result:

            # Plot the maximum likelihood
            ax1.plot(times, maxl, lw=0.5, color="C2", zorder=100)

            # Plot the 90%
            npreds = 100
            nsamples = len(result.posterior)
            preds = np.zeros((npreds, len(times)))
            for ii in range(npreds):
                draw = result.posterior.iloc[np.random.randint(nsamples)]
                preds[ii] = model(self.time, **draw)
            ax1.fill_between(
                times,
                np.quantile(preds, q=0.05, axis=0),
                np.quantile(preds, q=0.95, axis=0),
                color="C1", alpha=0.8, zorder=0)

            # Plot the sigma uncertainty on the residual
            median_sigma = np.median(result.posterior["sigma"])
            ax2.axhspan(-median_sigma, median_sigma, color='k', alpha=0.2)

            # Plot the 90% residual
            res_preds = self.flux - preds
            ax2.fill_between(
                times,
                np.quantile(res_preds, 0.05, axis=0),
                np.quantile(res_preds, 0.95, axis=0),
                color='C1', alpha=0.5, zorder=0)
            ax2.plot(times, self.flux - maxl, "C0", lw=0.5, zorder=100)

            # Auto-zoom to interesting region
            maxl_res = np.abs(maxl) - np.mean(result.posterior["base_flux"])
            maxl_peak = np.max(maxl_res)
            times_near_peak = times[maxl_res > 1e-3 * maxl_peak]
            if len(times_near_peak) > 1:
                ax1.set_xlim(np.min(times_near_peak), np.max(times_near_peak))

        ax1.set_ylabel("Flux")
        ax2.set_xlabel("Time - {} [days]".format(tref))
        ax2.set_ylabel("Flux residual")

        if result is not None:
            filename = "{}/{}_fit_with_data.png".format(outdir, label)
        else:
            filename = "{}/{}_data.png".format(outdir, label)

        fig.tight_layout()
        fig.savefig(filename, dpi=600)

    @property
    def normal_pvalue(self):
        return normaltest(self.flux).pvalue
