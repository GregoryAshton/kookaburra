Flux models
===========

Kookaburra is a tool to fit one-dimensional timeseries of flux data. We provide
a set of base class flux models which can be combined together and extended to
create arbitrarily complicated flux models.

Flux models
-----------

Each base flux model is initiated with a set of parameters related to its shape
(e.g., the maximum number of shapelets) and the suggested prior choices for
each parameter. The rational behind the prior recommentation is that it allows
the simple construction of priors for arbitrarily complicated models.

Shapelets
~~~~~~~~~

.. autoclass:: kookaburra.flux.ShapeleteFlux

Polynomials
~~~~~~~~~~~

.. autoclass:: kookaburra.flux.PolynomialFlux

Priors for a given flux model
-----------------------------

Once initiated, a flux model can provide a set of priors to use in analysing
some data. This requires the data to be provided to set things up like the
prior on the time of arrival. Here is an example for the shapelet model:

.. code-block:: python

   >>> import kookaburra as kb
   >>> shapelets = kb.flux.ShapeleteFlux(2)
   >>> # Create some random data
   >>> data = kb.data.TimeDomainData.from_array(
           time=np.linspace(0, 1, 100), flux=np.random.normal(0, 1, 100))
   >>> priors = shapelets.get_prior(data)
   >>> print(priors["toa"])
   Uniform(minimum=0.0, maximum=1.0, name='toa', latex_label='TOA', unit=None, boundary=None)

Combining flux models
---------------------

Once initiated, flux models can be combined. For example:

.. code-block:: python

   >>> import kookaburra as kb
   >>> shapelets = kb.flux.ShapeleteFlux(2)
   >>> poly = kb.flux.PolynomialFlux(2)
   >>> comibined  = shapelets + poly
   >>> print(combined.parameters)
   {'beta': None, 'toa': None, 'C0': None, 'C1': None, 'B0': None, 'B1': None}
