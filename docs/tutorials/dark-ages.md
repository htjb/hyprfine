# Calculating a Dark Ages 21-cm signal

The dark ages span roughly redshifts z ~ 30–200, corresponding to observed
frequencies of ~7–50 MHz. Over this period the 21-cm signal is set entirely
by the collisional coupling between the spin temperature and the gas
temperature — there are no stars yet, so the physics is clean and determined
solely by cosmology.

### 1. Define a frequency grid

`hyprfine` needs a frequency grid to evaluate the 21-cm signal over.

```python
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

# Frequencies in MHz covering the dark ages (z ~ 30-200)
f_grid = jnp.linspace(10, 50, 500)  # MHz
```

Note that `hyprfine` needs to be evaluated in float64.

### 2. Set up cosmological parameters

The `cosmology` namedtuple carries all the parameters needed by the signal
calculation. The fields are:

| Field | Description | Default value | Valid Range |
|-------|-------------|---------------|-------------|
| `H0` | Hubble constant (km/s/Mpc) | 67.36 | 20 - 100 |
| `Omega_b` | Baryon density | 0.049 | 0.005 - 0.1 |
| `Omega_c` | Cold dark matter density | 0.266 | 0.001 - 0.5 |
| `Y_He` | Helium mass fraction | 0.245 | 0.1 - 0.4 |
| `ns` | Scalar spectral index of primordial fluctuations | 0.97 | |
| `ln1010As` | $\ln 10^{10} A_s$ ; Amplitude of primordial power spectrum | 3.044 | |

Note that for the dark ages `ns` and `ln1010As` are not relevant, but they are used
for the cosmic dawn calculation to calculate the RMS matter density fluctuations and
mean star formation rate density. The default ranges on the other four parameters are defined by the HYREC-2 emulator training data.  

To initiate the cosmology parameters and change their values we write

```python
from hyprfine.parameters import cosmology

cosmo = cosmology(
    H0=70,
)
```

Any unspecified values are left at their defaults and indeed we can load in the
default set with `cosmo = cosmology()`.

Astrophysical parameters are defined with a `astrophysics()` object (see [here](cosmic-dawn.md)).
However, if the astrophysics object is not passed to `hyprfine` then astrophysics is ignored 
even at the redshifts where it is relevant.

### 3. Generate the signal

To generate the 21 cm signal we use the `generate_signal` function

```python
from hyprfine.analytic.main import generate_signal

T21, xe, Tk = generate_signal(f_grid, cosmo)
```

which returns the 21-cm brightness temperatures in mK, the free electron fraction
and the gas temperature in K all evaluated at `f_grid`.
