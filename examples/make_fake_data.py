""" Script to generate a data file with a single simulated pulse

Usage:

    $ python make_fake_data.py

This will generate a file fake_data.txt, a comma-separated file of the time
vs. flux for the simulated signal. The simulated signal contains a simple
first-order polynomial base flux and a three-component shapelet pulse.

"""
import numpy as np
import pandas as pd

import kookaburra as kb

# Injection parameters
pulse_injection_parameters = dict(
    C0=0.6, C1=0.1, C2=0.2, beta=1e-3, toa=0.005,  # Parameters for the shapelets
    B0=1, B1=-20  # Parameters for the polynomial base-flux
)


# Instantiate a flux model: a sum of the shaplet and polynomial flux classes
flux_model = kb.flux.ShapeletFlux(3) + kb.flux.PolynomialFlux(2)

# Generate fake data using the instantiated flux model and injection parameters
N = 1000
time = np.linspace(0, 2 * pulse_injection_parameters["toa"], N)
flux = flux_model(time, **pulse_injection_parameters)

# Add Gaussian noise
sigma = 2e-2
flux += np.random.normal(0, sigma, N)

# Write the data to a text file
df = pd.DataFrame(dict(time=time, flux=flux, pulse_number=0))
filename = "fake_data.txt"
df.to_csv(filename, index=False)
