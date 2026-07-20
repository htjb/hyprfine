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
 - name: Astrophysics Group, Cavendish Laboratory, University of Cambridge, J. J. Thomson Avenue, Cambridge, CB3 0US, UK 
   index: 1
 - name: Kavli Institute for Cosmology in Cambridge, University of Cambridge, Madingley Road, Cambridge, CB3 0HA, UK
   index: 2
date: 20 July 2026
bibliography: paper.bib
---

# Summary

# Statement of need

# State of the field

A number of other 21-cm simulation codes exist, and they fall roughly into three different classes; numerical, semi-numerical and analytic models. 

$C^2$-ray [@Mellema2006C2ray] is a numerical code written in Fortran90 focused on the Epoch of Reionization between redshifts $z=20$ and $z=6$. The radiative transfer algorithm uses ray-tracing to post process cosmological N-body simulations and calculate the ionization field from which the 21-cm signal can be calculated. The recent pyC$^2$ray upgrade uses a novel ray-tracing algorithm written in C++ with CUDA and a combination of Fortran90 and Python to compute the relevant chemistry and provide a user interface. In @Hirling2024pyc2ray the authors benchmarked pyC$^2$ray on an NVIDIA Tesla P100 and performed post-processing of a 349 MPc$^3$ N-body simulation in $\approx$ 2.5 GPU hours. The equivalent simulation using an older version of the code takes days to run on a CPU (13,824 core hours on 128 cores in parallel).

Two widely used semi-numerical codes are 21cmFAST and 21cmSPACE. 21cmFAST, like pyC$^2$ray is an open source code base written with Python and C.

# Software design

# Research impact statement

# AI usage disclosure

# Acknowledgements

# References