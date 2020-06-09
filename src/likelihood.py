import bilby
import numpy as np


class PulsarLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, data, model, noise_log_likelihood=np.nan):
        """
        A Gaussian likelihood for fitting pulsar flux data

        Parameters
        ----------
        data: kookaburra.data.TimeDomainData
            Object containing time and flux-density data to analyse
        model: kookaburra.flux.FluxModel
            The model to fit to the data
        """

        self.parameters = model.parameters
        self.parameters["sigma"] = None
        bilby.core.likelihood.Likelihood.__init__(self, dict.fromkeys(self.parameters))
        self._noise_log_likelihood = noise_log_likelihood

        self.x = data.time
        self.y = data.flux
        self.func = model

    def noise_log_likelihood(self):
        return self._noise_log_likelihood

    def log_likelihood(self):
        sigma = self.parameters["sigma"]
        residual = self.y - self.func(self.x, **self.parameters)
        log_l = np.sum(
            - .5 * ((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
        )
        return np.nan_to_num(log_l)
