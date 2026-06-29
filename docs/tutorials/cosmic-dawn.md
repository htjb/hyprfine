# Calculating a Cosmic Dawn 21-cm signal

The Cosmic Dawn covers roughly $z \sim 6$–50, corresponding to observed
frequencies of ~27–200 MHz. Over this period the first stars and galaxies
form, and their radiation heats the IGM via X-rays, couples the spin
temperature to the gas via the Wouthuysen-Field effect, and eventually
reionises the Universe. Modelling the signal therefore requires astrophysical
parameters on top of the cosmological ones.

### 1. Define a frequency grid

```python
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

# Frequencies in MHz covering the Cosmic Dawn (z ~ 6-50)
f_grid = jnp.linspace(30, 200, 500)  # MHz
```

### 2. Set up cosmological and astrophysical parameters

In addition to the `cosmology` namedtuple from the [Dark Ages tutorial](dark-ages.md),
the Cosmic Dawn calculation requires an `astrophysics` object. Its fields are:

| Field | Description | Default value |
|-------|-------------|---------------|
| `epsilon` | Star formation efficiency normalisation | 0.1 |
| `alpha_star` | Power-law slope of $f_*$ above pivot mass | 0.5 |
| `beta_star` | Power-law slope of $f_*$ below pivot mass | -0.5 |
| `M_pivot` | Pivot halo mass ($M_\odot$) | $3 \times 10^{11}$ |
| `L40` | X-ray luminosity per SFR ($10^{40}$ erg s $^{-1}$ / $M_\odot$ yr $^{-1}$) | 1.0 |
| `alpha_x` | X-ray spectral index | -1.0 |
| `nu_0` | X-ray lower band edge (keV) | 0.5 |
| `alpha_low` | Lyman-band SED slope below Ly-$\beta$ | 0.14 |
| `alpha_high` | Lyman-band SED slope above Ly-$\beta$ | -8.0 |
| `N_alpha` | Lyman photons per stellar baryon | 9690 |
| `f_esc` | Escape fraction of ionising photons | 0.15 |
| `N_ion` | Ionising photons per stellar baryon | 5000 |

```python
from hyprfine.parameters import cosmology, astrophysics

cosmo = cosmology()
astro = astrophysics()
```

Any unspecified fields take their default values.

### 3. Generate the signal

Pass the `astrophysics` object to `generate_signal` to activate the Cosmic Dawn
physics — the ODE solver for $T_k$ and $x_e$, X-ray heating, and the
Wouthuysen-Field coupling:

```python
from hyprfine.analytic.main import generate_signal

T21, xe, Tk = generate_signal(f_grid, cosmo, astro)
```

The function returns the 21-cm brightness temperature in mK, the free electron
fraction, and the gas temperature in K, all evaluated on `f_grid`.

### 4. Plot the result

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

axes[0].plot(f_grid, T21)
axes[0].set_ylabel(r"$T_{21}$ [mK]")
axes[0].grid()

axes[1].plot(f_grid, xe)
axes[1].set_ylabel(r"$x_e$")
axes[1].grid()

axes[2].plot(f_grid, Tk)
axes[2].set_ylabel(r"$T_k$ [K]")
axes[2].set_xlabel(r"$\nu$ [MHz]")
axes[2].grid()

plt.tight_layout()
plt.show()
```

### 5. Varying astrophysical parameters

You can explore how the signal changes with astrophysics by creating a new
`astrophysics` object with modified fields. For example, to increase the X-ray
heating efficiency:

```python
astro_hot = astrophysics(L40=10.0)

T21_hot, _, _ = generate_signal(f_grid, cosmo, astro_hot)

plt.plot(f_grid, T21, label="Default")
plt.plot(f_grid, T21_hot, label=r"$L_{40} = 10$")
plt.xlabel(r"$\nu$ [MHz]")
plt.ylabel(r"$T_{21}$ [mK]")
plt.legend()
plt.grid()
plt.show()
```

Note that the first call to `generate_signal` with astrophysics is slow due to
JAX JIT compilation. Subsequent calls with different parameters will be fast.
