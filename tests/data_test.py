import unittest

import numpy as np
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

    def test_max_time(self):
        data = TimeDomainData.from_array(time=self.time, flux=self.flux)
        self.assertAlmostEqual(data.max_time, 0, 3)


if __name__ == "__main__":
    unittest.main()
