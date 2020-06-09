import numpy as np


class SinglePulseFluxModel(object):
    """ Model-object for the flux

    Parameters
    ----------
    n_shapelets: int,
        The number of shapelets

    """

    def __init__(self, n_shapelets, n_base_flux):
        self.n_shapelets = n_shapelets
        self.n_base_flux = n_base_flux
        self._set_up_parameters()

    def _set_up_parameters(self):
        """ Initiates the parameters """
        self.amp_keys = ["C{}".format(i) for i in range(self.n_shapelets)]
        self.base_keys = ["B{}".format(i) for i in range(self.n_base_flux)]
        self.parameters = dict(beta=None, toa=None)
        for i in range(self.n_shapelets):
            self.parameters[self.amp_keys[i]] = None

    def get_base_flux(self, time, kwargs):
        coeffs = [kwargs[key] for key in self.base_keys]
        midtime = .5 * (time[0] + time[-1])
        return np.poly1d(coeffs[::-1])(time - midtime)

    def __call__(self, time, **kwargs):
        x = (time - kwargs["toa"]) / kwargs["beta"]
        pre = np.exp(-x ** 2)
        coefs = [kwargs[self.amp_keys[i]] for i in range(self.n_shapelets)]
        base_flux = self.get_base_flux(time, kwargs)
        return base_flux + pre * np.polynomial.hermite.Hermite(coefs)(x)
