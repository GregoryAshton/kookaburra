---
title: 'Kookaburra for radio-pulsar shapelet analysis'
tags:
  - Python
  - astronomy
  - radio pulsars
authors:
  - name: Gregory Ashton
    orcid: 0000-0001-7288-2231
    affiliation: "1, 2"
affiliations:
 - name: School of Physics and Astronomy, Monash University, VIC 3800, Australia
   index: 1
 - name: OzGrav, The ARC Centre of Excellence for Gravitational Wave Discovery, Clayton VIC 3800, Australia
   index: 2
date: 19 June 2020
bibliography: paper.bib
---

# Summary

Radio pulsars are rapidly-spinning highly-magnetized neutron stars which
produce a lighthouse-like beam of raditation. This radiation is observed by
radio telescopes as periodic pulsations. Kookaburra provides a method to fit
flux models to individual pulsations. Fitting is performed by stochastic
sampling methods building on the bilby [@bilby:2019] Bayesian inference
library.

# Shapelet model

The primary flux model provided by kookaburra is a simplified version of the
shapelet model proposed in @refregier:2003. We define shapelets as

$$ f(x) = \sum_{i=0}^{n_{s}} C_{i} H_{i}(x/\beta) e^{-x^2 / \beta^2} \,, $$

where $C_{i}$ are the shapelet coefficients, $H_{i}$ is the Hermite polynomial
of degree $i$, and $\beta$ is a width parameter. The complete flux model fit to
the data is then $f(x - t)$ where $t$ is the pulse time of arrival.
