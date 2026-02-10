import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import cosmology

z_init = 1100
z_grid = jnp.linspace(10, z_init, 500)
f_grid = 1420.4 / (z_grid + 1)

sample = jnp.array(
    [67.36, 0.315, 0.049, 0.266, 0.245]
)  # Example cosmological parameters


cosmo = cosmology(
    H0=67.36,
    Omega_m=0.315,
    Omega_b=0.049,
    Omega_c=0.266,
    Y_He=0.245,
    ns=0.97,
    ln1010As=3.1,
)  # Example cosmology parameters

T21_values = generate_signal(f_grid, cosmo, z_init)

plt.plot(z_grid, T21_values)
plt.xscale("log")
plt.xlabel("Redshift z")
plt.ylabel("21cm Brightness Temperature T21 (mK)")
plt.title("21cm Signal vs Redshift")
plt.grid()
plt.show()
