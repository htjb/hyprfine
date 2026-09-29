---
title: 'hyprfine: simulating the 21-cm signal from the Dark Ages through to the Epoch of Reionization on a GPU'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Harry T. J. Bevins
    orcid: 0000-0002-4367-3550
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physics, Imperial College London, Blackett Laboratory, Prince Consort Road, London SW7 2AZ, UK
   index: 1
 - name: I-X Centre for AI in Science, Translation and Innovation Hub (I-HUB), Imperial College London, White City Campus, 84 Wood Lane, London W12 0BZ, UK
   index: 2
date: 20 July 2026
bibliography: paper.bib
---

# Summary

We present `hyprfine`, a GPU accelerated analytic model of the sky-averaged 21-cm signal from the Dark Ages through to the Cosmic Dawn between approximately $z=1100$ and $z=5$. The 21-cm signal arises from the spin flip transition in neutral hydrogen and the relative number of atoms in each spin state can be characterized by a statistical temperature known as the spin temperature. The relative number of atoms in each state and the spin temperature is driven by different processes over cosmic time including interactions with the radio background, collisions between different hydrogen atoms, light from the first galaxies, X-ray emission from the early universe, and ionizing UV emission. The signal is measured relative to the radio background $\delta T_b \propto (1 - \frac{T_\gamma}{T_s})$.

As far as the authors are aware `hyprfine` is the first analytic model of the 21-cm signal written in JAX, the first that can be run natively on GPUs and the first to take advantage of autodiff to differentiate the dark ages 21-cm signal between approximately $z=300$ and $z=50$ with respect to the cosmological parameters of the model.

# Statement of need

# State of the field

A number of other 21-cm simulation codes exist, and they fall roughly into three different classes; numerical, semi-numerical and analytic models. 

$C^2$-ray [@Mellema2006C2ray] is a numerical code written in Fortran90 focused on the Epoch of Reionization between redshifts $z=20$ and $z=6$. The radiative transfer algorithm uses ray-tracing to post process cosmological N-body simulations and calculate the ionization field from which the 21-cm signal can be calculated. The recent pyC$^2$ray upgrade uses a novel ray-tracing algorithm written in C++ with CUDA and a combination of Fortran90 and Python to compute the relevant chemistry and provide a user interface. In @Hirling2024pyc2ray the authors benchmarked pyC$^2$ray on an NVIDIA Tesla P100 and performed post-processing of a 349 MPc$^3$ N-body simulation in $\approx$ 2.5 GPU hours. The equivalent simulation using an older version of the code takes days to run on a CPU (13,824 core hours on 128 cores in parallel).

Two widely used semi-numerical codes are 21cmFAST and 21cmSPACE. 21cmFAST, like pyC$^2$ray is an open source code base written with Python and C.

hyprfine is an analytic code and is most similar to zeus21 and ECHO21.

![\textbf{Left:} An example 21-cm signal from hyprfine and zeus21 with the same paraemters. \textbf{Right:} The run time of hyprfine per signal on an AMD Ryzen 5, a Tesla T4 and an A100 compared to the value for zeus21. \label{fig:benchmark}](bin/benchmark.png)

| Code | Simulation Type | Products | Approx. Redshift | GPU? | Programming Lang./Framework | Open Source |
|------|-----------------|----------|------------------|------|-----------------------------|-------------|
| pyC$^2$ray | Numerical |  Spatially resolved ionization field | 21 - 6 |Yes (partial) | Python, Fortran90, C++, CUDA | Yes |
| 21cmFAST | Semi-numerical | 21-cm simulation box, summary statistics and more | 50 - 6 | Yes (partial) \textbf{Need to check this!} | Python, C | Yes |
| 21cmSPACE | Semi-numerical | 21-cm simulation box, summary statistics and more | 50 - 6 | No | Matlab, Python | No |
| Zeus21 | Analytic | 21-cm global signal, power spectrum, UVLF | 35 - 5 | No | Python | Yes |
| ECHO21 | Analytic | 21-cm global signal | 1500 - 0 | No | Python | Yes |
| ARES | Semi-analytic/1D RT | 21-cm global signal and more | 35 - 5 | No | Python | Yes |
| hyprfine | Analytic | 21-cm global signal | 1100 - 0 | Yes | Python, JAX | Yes |


# Software design

# Research impact statement

# AI usage disclosure

# Acknowledgements

# References