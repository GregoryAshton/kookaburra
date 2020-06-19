import numpy as np
from bilby.core.prior import PriorDict, Uniform, LogUniform, Normal

from .priors import SpikeAndSlab


class JointFluxModel(object):
    def __init__(self, flux_model_A, flux_model_B):
        self.models = []

        if isinstance(flux_model_A, JointFluxModel):
            self.models += flux_model_A.models
        elif isinstance(flux_model_A, BaseFlux):
            self.models.append(flux_model_A)
        else:
            raise ValueError()

        if isinstance(flux_model_B, JointFluxModel):
            self.models += flux_model_B.models
        elif isinstance(flux_model_B, BaseFlux):
            self.models.append(flux_model_B)
        else:
            raise ValueError()

    @property
    def parameters(self):
        p = dict()
        for model in self.models:
            p.update(model.parameters)
        return p

    def get_priors(self, data):
        priors = PriorDict()
        for model in self.models:
            priors.update(model.get_priors(data))
        return priors

    def __call__(self, time, **kwargs):
        fluxes = [model(time, **kwargs) for model in self.models]
        return np.sum(fluxes, axis=0)

    def get_pulse_only(self, time, **kwargs):
        fluxes = []
        for model in self.models:
            if model.pulse:
                fluxes.append(model(time, **kwargs))
        return np.sum(fluxes, axis=0)

    def __add__(self, flux_model):
        return JointFluxModel(self, flux_model)


class BaseFlux(object):
    """ A base flux model to be overwritten """
    pulse = False

    def __init__(self):
        self.parameters = dict()

    def get_priors(self, data):
        return PriorDict()

    def __add__(self, flux_model):
        return JointFluxModel(self, flux_model)

    def __call__(self, time, **kwargs):
        return np.zeros_like(time)


class ShapeletFlux(BaseFlux):
    r""" An arbitrary shapelet basis

    We use a simplified version of the shapelet formalism. Specifically, the
    flux is modelled as

    .. math::
        f(t, n_s) = \sum_{i=0}^{n_s} C_{i} H_{i}\left(\frac{t-t_0}{\beta}\right) e^{\frac{(t-t_0)^2}{\beta^2}}

    where :math:`C_{i}` are the coefficients, :math:`H_{i}` is the Hermite
    polynomial of degree :math:`i`, and :math:`t_0` is the time of arrival.

    Parameters
    ----------
    n_shapelets: int,
        The number of shapelets
    parameter_base_name: str
        The basename for the coefficients, defaults to "C" such that
        coefficients are "C0", "C1", etc.
    toa_prior_time: str, float
        If "auto" (default), the middle of the toa prior is chosen
        automatically based on the maximum in the flux (with some robustness
        checks). If a float [0, 1] it is taken as a fraction through the
        duration at which to set the centre time, e.g. 0.5 specifies a prior
        centered in the middle of the data segment. If toa_prior_width=1, this
        parameter is ignored.
    toa_prior_width: float
        The fraction (of the duration) for the prior width. If equal to 1, the
        entire data span is used as a prior. If less than one, a uniform prior
        centered on toa_prior_time with with duration * toa_prior_width is used.
    c_mix: float
        The prior mixture fraction between the slab and spike for the
        shapelet coefficients.
    c_max_multiplier: float
        A muliplier on the maximum coefficient amplitude to use for the slab
        prior.
    beta_min, beta_max: float, None
        The minimum and maximum value for the beta prior. If None, default
        default values are chosen based on the data duration and sample rate.
    beta_type: str [uniform, log-uniform]
        The type of prior to use for the beta parameter

    """
    pulse = True

    def __init__(self, n_shapelets, parameter_base_name="C", toa_prior_width=1,
                 toa_prior_time=None, c_mix=0.5, c_max_multiplier=1,
                 beta_min=None, beta_max=None, beta_type="uniform"):
        self.n_shapelets = n_shapelets
        self.parameter_base_name = parameter_base_name

        # Prior arguments
        self.toa_prior_width = toa_prior_width
        self.toa_prior_time = toa_prior_time
        self.c_mix = c_mix
        self.c_max_multiplier = c_max_multiplier
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_type = beta_type

        # Set up the keys and parameters
        self.keys = [
            f"{self.parameter_base_name}{ii}" for ii in range(self.n_shapelets)
        ]
        self.parameters = dict(beta=None, toa=None)
        for i in range(self.n_shapelets):
            self.parameters[self.keys[i]] = None

    def __call__(self, time, **kwargs):
        """ Return the flux as a function of time """
        x = (time - kwargs["toa"]) / kwargs["beta"]
        pre = np.exp(-(x ** 2))
        coefs = [kwargs[key] for key in self.keys]
        return pre * np.polynomial.hermite.Hermite(coefs)(x)

    def get_priors(self, data):
        priors = PriorDict()

        # Set up the TOA prior
        if self.toa_prior_width < 1:
            if self.toa_prior_time == "auto":
                t0 = data.estimate_pulse_time()
            else:
                t0 = data.start + float(self.toa_prior_time) * data.duration
            dt = data.duration * self.toa_prior_width
            priors["toa"] = Uniform(t0 - dt, t0 + dt, "toa", latex_label="TOA")
        else:
            priors["toa"] = Uniform(data.start, data.end, "toa", latex_label="TOA")

        # Set up the beta prior
        if self.beta_min is None:
            self.beta_min = data.time_step
        if self.beta_max is None:
            self.beta_max = data.duration

        if self.beta_type == "uniform":
            priors["beta"] = Uniform(self.beta_min, self.beta_max, "beta", latex_label=r"$\beta$")
        elif self.beta_type == "log-uniform":
            priors["beta"] = LogUniform(
                self.beta_min, self.beta_max, "beta", latex_label=r"$\beta$"
            )
        else:
            raise ValueError()

        # Set up the coefficient prior
        for key in self.keys:
            priors[key] = SpikeAndSlab(
                slab=Uniform(1e-20 * data.max_flux, self.c_max_multiplier * data.range_flux),
                name=key,
                mix=self.c_mix,
            )
        return priors


class PolynomialFlux(BaseFlux):
    """ An arbitrary-degree polynomial: used for modelling the baseline flux

    The flux is given as the sum of polynomials up to :code:`n_polynomials`.
    All coefficients are applied with a reference time in the middle of the
    observation span.

    Parameters
    ----------
    n_polynomials: int,
        The number of shapelets
    parameter_base_name: str
        The basename for the coefficients, defaults to "C" such that
        coefficients are "C0", "C1", etc.

    """
    pulse = False

    def __init__(self, n_polynomials, parameter_base_name="B"):
        self.n_polynomials = n_polynomials
        self.parameter_base_name = parameter_base_name
        self._set_up_parameters()

    def _set_up_parameters(self):
        """ Initiates the parameters """
        self.keys = [
            f"{self.parameter_base_name}{ii}" for ii in range(self.n_polynomials)
        ]

        self.parameters = {key: None for key in self.keys}

    def __call__(self, time, **kwargs):
        coeffs = [kwargs[key] for key in self.keys]
        midtime = 0.5 * (time[0] + time[-1])
        return np.poly1d(coeffs[::-1])(time - midtime)

    def get_priors(self, data):
        priors = PriorDict()

        name = f'{self.parameter_base_name}0'
        priors[name] = Uniform(
            0, data.max_flux, name, latex_label='$B_{0}$')
        for ii in range(1, self.n_polynomials):
            name = f'{self.parameter_base_name}{ii}'
            priors[name] = Normal(
                0,
                data.range_flux / data.duration ** ii / np.math.factorial(ii),
                name,
                latex_label=f'$B_{{{ii}}}$')
        return priors
