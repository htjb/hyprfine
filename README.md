# hyprfine

**Authors:** Harry T. J. Bevins <br>
**Version**: 0.1.0 <br>
**Homepage**: [https://github.com/htjb/hyprfine](https://github.com/htjb/hyprfine)<br>
**License**: MIT

GPU-accelerated simulation tools for the cosmological 21-cm signal, built on [JAX](https://github.com/google/jax). hyprfine replaces slow C-code recombination solvers with neural network emulators, enabling fast, differentiable signal generation suitable for inference pipelines.

![Benchmark](benchmark.png)

*Dark ages 21-cm signal computed with Planck 2018 parameters (left) and wall-clock timing on CPU (right). The warm (post-JIT) evaluation takes ~2 ms.*

## Installation

```bash
pip install .
```

For development:

```bash
pip install ".[dev]"
pre-commit install
```

## Quick start

```python
import jax.numpy as jnp
from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import cosmology

cosmo = cosmology(
    H0=67.32, Omega_m=0.3158, Omega_b=0.0494, Omega_c=0.2664, Y_He=0.2454
)
f_grid = jnp.linspace(10.0, 100.0, 500)  # MHz
T21 = generate_signal(f_grid, cosmo)
```

## Documentation

Full documentation can be built and served locally:

```bash
pip install ".[docs]"
mkdocs serve
```

## Citation

If you use hyprfine in your research please link to this repo. 