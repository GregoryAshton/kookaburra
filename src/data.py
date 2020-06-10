import numpy as np
import pandas as pd


class TimeDomainData:
    """ Object to store time-domain flux data """

    def __init__(self, time_unit="days"):
        self.time_unit = time_unit

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
    def min_flux(self):
        """ Return the maximum flux """
        return np.min(self.flux)

    @property
    def range_flux(self):
        """ Return the maximum flux """
        return self.max_flux - self.min_flux

    @property
    def max_time(self):
        """ Return the time of the maximum flux """
        return self.time[np.argmax(self.flux)]

    @property
    def midtime(self):
        return .5 * (self.start + self.end)

    @property
    def time_step(self):
        return self.time[1] - self.time[0]

    @property
    def reference_time(self):
        """ Return the reference time, defaults to the mid time """
        if hasattr(self, '_reference_time') is False:
            self._reference_time = self.midtime

        return self._reference_time

    @property
    def delta_time(self):
        """ Return time - reference_time """
        return self.time - self.reference_time

    @reference_time.setter
    def reference_time(self, reference_time):
        self._reference_time = reference_time

    @property
    def time_unit(self):
        return self._time_unit

    @time_unit.setter
    def time_unit(self, time_unit):
        self._time_unit = time_unit

    def truncate_data(self, width):
        idxs = np.abs(self.delta_time) < width * self.duration
        self.time = self.time[idxs]
        self.flux = self.flux[idxs]

    def estimate_pulse_time(self, f=0.75):
        """ Naive estimate of the pulse time

        Uses the mean of flux above a fraction f of the maximum fluc

        Parameters
        ----------
        f: float
            The fraction to use

        Returns
        -------
        pulse_time: float
            An estimate of the pulse time
        """
        idxs = np.abs(self.flux) > f * self.max_flux
        return np.mean(self.time[idxs])

    @classmethod
    def from_array(cls, time, flux):
        """ Read in the time and flux from a csv

        Parameters
        ----------
        time, flux: np.ndarray
            The time and flux arrays

        """

        time = np.atleast_1d(time)
        flux = np.atleast_1d(flux)

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

        The filename must point to a comma-separated file with at least two
        columns, "time" and "flux". Optionally, an additional "pulse_number"
        column can exist, if the pulse_number is specified, only data matching
        the requested pulse number will be loaded.

        Parameters
        ----------
        filename: str
            The path to the file to read
        pulse_number: int:
            The pulse number to select from the file: if this is not given the
            entire data file is used.
        """
        df = pd.read_csv(filename)
        return cls._sort_and_filter_dataframe(df, pulse_number)

    @classmethod
    def from_file(cls, filename, pulse_number=None):
        """ Read in the time and flux from a file

        This is a generic interface for any file, for information about the
        allowed filetypes and their specification, see the from_filetype
        methods.

        Parameters
        ----------
        filename: str
            The path to the file to read
        pulse_number: int:
            The pulse number to select from the file: if this is not given the
            entire data file is used.

        """
        if "h5" in filename:
            return cls.from_h5(filename, pulse_number)
        else:
            return cls.from_csv(filename, pulse_number)

    @classmethod
    def from_h5(cls, filename, pulse_number=None):
        """ Read in the time and flux from a pandas h5 file

        The filename must point to a h5 data-frame file. The dataframe should
        have at least two columns, "time" and "flux". Optionally, an additional
        "pulse_number" column can exist, if the pulse_number is specified,
        only data matching the requested pulse number will be loaded.

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
