import unittest
import os

import numpy as np
import pandas as pd

from kookaburra.data import TimeDomainData


class Data(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(-1, 1, 10000)
        self.flux = np.exp(-self.time ** 2 / 2)

    def tearDown(self):
        del self.time
        del self.flux

    def test_duration(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.duration, self.time[-1] - self.time[0])

    def test_start(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.start, self.time[0])

    def test_end(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.end, self.time[-1])

    def test_N(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.N, len(self.time))

    def test_max_flux(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertAlmostEqual(data.max_flux, np.exp(0), 3)

    def test_min_flux(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.min_flux, np.min(data.flux))

    def test_range_flux(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.range_flux, np.max(self.flux) - np.min(self.flux))

    def test_midtime(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.midtime, 0)

    def test_reference_time_default(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.reference_time, 0)

    def test_reference_time_set(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        data.reference_time = 0.3
        self.assertEqual(data.reference_time, 0.3)

    def test_max_time(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertAlmostEqual(data.max_time, 0, 3)

    def test_delta_time(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        data.reference_time = 10
        delta_time = data.time - 10
        self.assertTrue(all(data.delta_time - delta_time == 0))

    def test_RMS_flux(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.RMS_flux, np.sqrt(np.mean(data.flux ** 2)))

    def test_estimate_pulse_time(self):
        passes = 0
        total = 10
        time = np.linspace(-1, 1, 1000)
        for _ in range(total):
            pulse_time = np.random.uniform(-0.5, 0.5)
            pulse_width = 0.01
            noise = np.random.normal(0, 1, len(time))
            flux = 4 * np.exp(-(time - pulse_time) ** 2 / pulse_width) + noise
            data = TimeDomainData.from_array(time=time, flux=flux)
            if np.abs(data.estimate_pulse_time() - pulse_time) < 0.1:
                passes += 1

        self.assertTrue(passes == total)

    def test_from_array(self):
        data = TimeDomainData.from_array(self.time, self.flux)
        self.assertTrue(all(self.time == data.time))
        self.assertTrue(all(self.flux == data.flux))

    def test_from_array_unequal_shape(self):
        with self.assertRaises(ValueError):
            TimeDomainData.from_array(self.time[:10], self.flux)

    def test_from_array_ndim_greater_one(self):
        with self.assertRaises(ValueError):
            TimeDomainData.from_array(
                np.random.uniform(0, 1, (2, 3)),
                np.random.uniform(0, 1, (2, 3)))

    def test_from_list(self):
        time = [0, 1, 2]
        flux = [0, 1, 2]
        TimeDomainData.from_array(time, flux)

    def test_from_unsorted(self):
        time = [0, 1, 2, 1.5]
        flux = [0, 1, 2, 1.5]
        with self.assertRaises(ValueError):
            TimeDomainData.from_array(time, flux)

    def test_read_csv(self):
        df = pd.DataFrame(dict(time=self.time, flux=self.flux, pulse_number=0))
        filename = "testing.csv"
        df.to_csv(filename, float_format='%.16f')
        data = TimeDomainData.from_csv(filename)
        os.remove(filename)
        self.assertTrue(np.all(np.abs(data.time - df.time.values) < 1e-15))

    def test_read_h5(self):
        df = pd.DataFrame(dict(time=self.time, flux=self.flux, pulse_number=0))
        filename = "testing.h5"
        df.to_hdf(filename, 'df')
        data = TimeDomainData.from_h5(filename)
        os.remove(filename)
        self.assertTrue(np.all(np.abs(data.time - df.time.values) < 1e-15))

    def test_from_file(self):
        df = pd.DataFrame(dict(time=self.time, flux=self.flux, pulse_number=0))
        h5_filename = "testing.h5"
        df.to_hdf(h5_filename, 'df')

        csv_filename = "testing.csv"
        df.to_csv(csv_filename)

        h5data = TimeDomainData.from_file(h5_filename)
        csvdata = TimeDomainData.from_file(csv_filename)
        os.remove(h5_filename)
        os.remove(csv_filename)
        self.assertTrue(all(np.abs(h5data.flux - csvdata.flux) < 1e-15))
        self.assertTrue(all(np.abs(h5data.time - csvdata.time) < 1e-15))

    def test_pulse_number(self):
        time = np.linspace(0, 3, 1000)
        flux = np.random.normal(0, 1, len(time))
        pulse_number = np.zeros(len(time))
        pulse_number[time > 1] = 1
        pulse_number[time > 2] = 2
        df = pd.DataFrame(dict(time=time, flux=flux, pulse_number=pulse_number))

        filename = "testing.csv"
        df.to_csv(filename, float_format='%.16f')
        data = TimeDomainData.from_csv(filename, pulse_number=0)
        os.remove(filename)
        idxs = df.pulse_number == 0
        self.assertTrue(np.all(np.abs(data.time - df[idxs].time.values) < 1e-15))

    def test_time_unit_default(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertEqual(data.time_unit, "days")

    def test_time_unit_set(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        data.time_unit = "s"
        self.assertEqual(data.time_unit, "s")

    def test_normal_pvalue(self):
        data = TimeDomainData.from_array(
            time=self.time, flux=np.random.normal(0, 1, len(self.time)))
        self.assertGreater(data.normal_pvalue, 1e-2)


if __name__ == "__main__":
    unittest.main()
