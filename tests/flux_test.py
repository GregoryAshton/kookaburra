import unittest

import numpy as np
import bilby

from kookaburra import flux
from kookaburra.data import TimeDomainData


class Flux(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(-1, 1, 10000)
        self.flux = np.exp(-self.time ** 2 / 2)
        self.data = TimeDomainData.from_array(time=self.time, flux=self.flux)

    def tearDown(self):
        del self.time
        del self.flux

    def test_shapelets(self):
        flux_instance = flux.ShapeleteFlux(3)
        self.assertIsInstance(flux_instance.parameters, dict)
        self.assertEqual(list(flux_instance.parameters.keys()),
                         ["beta", "toa", "C0", "C1", "C2"])
        priors = flux_instance.get_priors(self.data)
        self.assertIsInstance(priors, bilby.core.prior.PriorDict)
        out = flux_instance(self.data.time, **priors.sample())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.time.shape)

    def test_shapelets_toa_prior_width(self):
        flux_instance = flux.ShapeleteFlux(5, toa_prior_width=0.1, toa_prior_time=0.1)
        priors = flux_instance.get_priors(self.data)
        self.assertIsInstance(priors, bilby.core.prior.PriorDict)
        self.assertLess(priors["toa"].maximum - priors["toa"].minimum, self.data.duration)

    def test_shapelets_beta_uniform_prior(self):
        flux_instance = flux.ShapeleteFlux(5, beta_type="uniform", beta_min=0, beta_max=0.1)
        priors = flux_instance.get_priors(self.data)
        self.assertIsInstance(priors["beta"], bilby.core.prior.Uniform)
        self.assertEqual(priors["beta"].minimum, 0)
        self.assertEqual(priors["beta"].maximum, 0.1)

    def test_shapelets_beta_log_uniform_prior(self):
        flux_instance = flux.ShapeleteFlux(5, beta_type="log-uniform", beta_min=0.01, beta_max=0.1)
        priors = flux_instance.get_priors(self.data)
        self.assertIsInstance(priors["beta"], bilby.core.prior.LogUniform)
        self.assertEqual(priors["beta"].minimum, 0.01)
        self.assertEqual(priors["beta"].maximum, 0.1)

    def test_polynomial(self):
        flux_instance = flux.PolynomialFlux(3)
        self.assertIsInstance(flux_instance.parameters, dict)
        self.assertEqual(list(flux_instance.parameters.keys()),
                         ["B0", "B1", "B2"])
        priors = flux_instance.get_priors(self.data)
        self.assertIsInstance(priors, bilby.core.prior.PriorDict)
        out = flux_instance(self.data.time, **priors.sample())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.time.shape)

    def test_shapelets_and_polynomial(self):
        flux_instance = flux.PolynomialFlux(5) + flux.ShapeleteFlux(3)
        self.assertIsInstance(flux_instance.parameters, dict)
        priors = flux_instance.get_priors(self.data)
        self.assertIsInstance(priors, bilby.core.prior.PriorDict)
        out = flux_instance(self.data.time, **priors.sample())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.time.shape)


if __name__ == "__main__":
    unittest.main()
