"""Test the Wouthuysen-Field coupling calculations."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import astrophysics, cosmology

cosmo = cosmology(
    H0=67.36,
    Omega_m=0.315,
    Omega_b=0.049,
    Omega_c=0.266,
    Y_He=0.245,
    ns=0.97,
    ln1010As=3.044,
)  # Example cosmology parameters

astro = astrophysics(
    epsilon=0.1,
    alpha_star=0.5,
    beta_star=-0.5,
    M_pivot=3e11,
)

T21_values, xe, T_gas, xc_values, T_s, T_cmb, xalpha_values = generate_signal(
    f_grid=jnp.linspace(5, 200, 100),  # Frequency grid in MHz
    cosmo=cosmo,
    astro=astro,
    detailed_output=True,
)

plt.plot(jnp.linspace(5, 200, 100), T21_values)
plt.xlabel("Frequency (MHz)")
plt.ylabel("21cm brightness temperature T21 (mK)")
plt.title("21cm brightness temperature T21 vs frequency")
plt.grid()
plt.savefig("T21_vs_frequency.png")
plt.show()

freq = jnp.linspace(5, 200, 100)
plt.plot(freq, T_cmb, label="T_cmb")
plt.plot(freq, T_gas, label="T_gas")
plt.plot(freq, T_s, label="T_s")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Temperature (K)")
plt.title("Temperatures vs frequency")
plt.loglog()
plt.legend()
plt.grid()
plt.savefig("temperatures_vs_frequency.png")
plt.show()
