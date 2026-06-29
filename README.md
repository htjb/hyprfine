# hyprfine

**Authors:** Harry T. J. Bevins <br>
**Version**: 1.0.1 <br>
**Homepage**: [https://github.com/htjb/hyprfine](https://github.com/htjb/hyprfine)<br>
**License**: MIT

GPU-accelerated simulation tools for the cosmological 21-cm signal from the Dark Ages (DA), Cosmic Dawn (CD) and Epoch of Reionization (EoR). `hyprfine` is built using [JAX](https://github.com/google/jax) with the aim of building a signal model that runs on the GPU and is differentiable.

Currently, the code features an analytic model for the sky-averaged 21-cm signal from the dark ages through to low redshifts at the end of the EoR. At the moment only the dark ages part of the code is differentiable. The graph below is generated with the code in `bin/benchmark_time.py` and `bin/benchmark_plot.py`. It shows the relative performance on CPU and GPU as a function of batch size as well as a comparison with [zeus21](https://github.com/ZeusCosmo/Zeus21). 

![Benchmark](https://raw.githubusercontent.com/htjb/hyprfine/main/bin/benchmark.png)


## Installation

The code is pip installable with `pip install hyprfine` but for the latest version it is best to git clone the repo and install

```bash
git clone https://github.com/htjb/hyprfine
cd hyprfine
pip install .
```

For development:

```bash
pip install ".[dev]"
pre-commit install
```

## Quick start

To generate a signal with the default parameters you can run

```python
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import cosmology, astrophysics

f_grid = jnp.linspace(10.0, 100.0, 500)  # MHz
T21, xe, Tk = generate_signal(f_grid, cosmology(), astrophysics())
```

To change the default parameters you can specify

```python
cosmo = cosmology(
    H0=67.36, # hubble's constant
    Omega_b=0.049, # Baryon density
    Omega_c=0.266, # Dark matter density
)

astro = astrophysics(
    epsilon=0.1, # normalisation of star formation efficiency
    alpha_star=0.5, # slope of low mass star formation efficiency model
    beta_star=-0.5, # slope of high mass sfe model
    M_pivot=3e11, # pivot mass of sfe model
)
```

for a full list of tunable parameters see `hyprfine.parameters`.

To take the derivatives across the Dark Ages global signal model you can run

```python
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import astrophysics, cosmology

fgrid = jnp.linspace(5, 50, 100)  # Frequency grid in MHz
dT21dcosmo, dxedcosmo, dTgasdcosmo = jax.jacfwd(generate_signal, argnums=1)(
    fgrid, cosmo
)

dT21dOmega_b = dT21dcosmo.Omega_b
dxedH0 = dxedcosmo.H0 # etc...
```

## Documentation

Full documentation including tutorials can be built and served locally:

```bash
pip install ".[docs]"
mkdocs serve
```

## Citation

If you use `hyprfine` in your research please link to this repo. 