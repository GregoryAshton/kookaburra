import bilby
import numpy as np


class NullLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, data, model):
        """
        A Gaussian likelihood for fitting a null to the flux data

        Parameters
        ----------
        data: pyglit.main.TimeDomainData
            Object containing time and flux-density data to analyse
        """

        self.parameters = dict(sigma=None)
        self.parameters.update({key: None for key in model.base_keys})
        bilby.core.likelihood.Likelihood.__init__(self, dict.fromkeys(self.parameters))

        self.x = data.time
        self.y = data.flux
        self.model = model

    def log_likelihood(self):
        sigma = self.parameters["sigma"]
        base_flux = self.model.get_base_flux(self.x, self.parameters)
        residual = self.y - base_flux
        log_l = np.sum(
            - .5 * ((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
        )
        return log_l


class PulsarLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, data, model):
        """
        A Gaussian likelihood for fitting pulsar flux data

        Parameters
        ----------
        data: pyglit.main.TimeDomainData
            Object containing time and flux-density data to analyse
        model: pyglit.main.PulsarFluxModel
            The model to fit to the data
        """

        self.parameters = model.parameters
        self.parameters["sigma"] = None
        bilby.core.likelihood.Likelihood.__init__(self, dict.fromkeys(self.parameters))

        self.x = data.time
        self.y = data.flux
        self.func = model

    def log_likelihood(self):
        sigma = self.parameters["sigma"]
        residual = self.y - self.func(self.x, **self.parameters)
        log_l = np.sum(
            - .5 * ((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
        )
        return np.nan_to_num(log_l)
