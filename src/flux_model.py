import numpy as np


class SinglePulseFluxModel(object):
    """ Model-object for the flux

    Parameters
    ----------
    n_shapelets: int,
        The number of shapelets

    """

    def __init__(self, n_shapelets):
        self.n_shapelets = n_shapelets
        self._set_up_parameters()

    def _set_up_parameters(self):
        """ Initiates the parameters """
        self.amp_keys = ["C{}".format(i) for i in range(self.n_shapelets)]
        self.parameters = dict(beta=None, toa=None, base_flux=None)
        for i in range(self.n_shapelets):
            self.parameters[self.amp_keys[i]] = None

    def __call__(self, time, **kwargs):
        x = (time - kwargs["toa"]) / kwargs["beta"]
        pre = np.exp(-x ** 2)
        coefs = [kwargs[self.amp_keys[i]] for i in range(self.n_shapelets)]
        return kwargs["base_flux"] + pre * np.polynomial.hermite.Hermite(coefs)(x)
