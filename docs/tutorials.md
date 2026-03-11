# Tutorials

## Calculating a Dark Ages 21-cm signal

The dark ages span roughly redshifts z ~ 30–200, corresponding to observed
frequencies of ~7–50 MHz. Over this period the 21-cm signal is set entirely
by the collisional coupling between the spin temperature and the gas
temperature — there are no stars yet, so the physics is clean and determined
solely by cosmology.

### 1. Define a frequency grid

```python
import jax.numpy as jnp

# Frequencies in MHz covering the dark ages (z ~ 30-200)
f_grid = jnp.linspace(10, 50, 500)  # MHz
```

### 2. Set up cosmological parameters

The `cosmology` namedtuple carries all the parameters needed by the signal
calculation. The fields are:

| Field | Description | Typical value |
|-------|-------------|---------------|
| `H0` | Hubble constant (km/s/Mpc) | 67.32 |
| `Omega_m` | Total matter density | 0.3158 |
| `Omega_b` | Baryon density | 0.0494 |
| `Omega_c` | Cold dark matter density | 0.2664 |
| `Y_He` | Helium mass fraction | 0.2454 |

```python
from hyprfine.parameters import cosmology

cosmo = cosmology(
    H0=67.32,
    Omega_m=0.3158,
    Omega_b=0.0494,
    Omega_c=0.2664,
    Y_He=0.2454,
)
```

### 3. Generate the signal

```python
from hyprfine.analytic.main import generate_signal

T21 = generate_signal(f_grid, cosmo)
```

Returns a JAX array of 21-cm brightness temperatures in mK, one value per
frequency in `f_grid`. The redshift range is set by the training data of the
recombination emulator.

### 4. Inspect the output

```python
import matplotlib.pyplot as plt

plt.plot(f_grid, T21)
plt.xlabel("Frequency (MHz)")
plt.ylabel("$T_{21}$ (mK)")
plt.title("Dark Ages 21-cm signal")
plt.show()
```

### Detailed output

Pass `detailed_output=True` to also retrieve the intermediate quantities at
each redshift:

```python
T21, xe, Tk, xc = generate_signal(f_grid, cosmo, detailed_output=True)
```

| Variable | Description |
|----------|-------------|
| `T21` | Brightness temperature (mK) |
| `xe` | Free electron fraction |
| `Tk` | Gas kinetic temperature (K) |
| `xc` | Collisional coupling coefficient |
