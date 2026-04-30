"""Test the Wouthuysen-Field coupling calculations."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.wouthuysen_field import ( 
    calculate_epsilon_alpha_tot,
)

lyman_alpha_frequency = 2.466e15  # Hz
lyman_beta_frequency = 2.922e15  # Hz
lyman_limit = 3.289e15  # Hz


z = [10, 12, 15, 20, 25, 30]

nu = 10 ** jnp.arange(
    jnp.log10(lyman_alpha_frequency), jnp.log10(lyman_limit), 0.001
)

for z in z:
    epsilon = calculate_epsilon_alpha_tot(z=z, z_source=20)
    nu_prime = nu * (1 + 20) / (1 + z)
    plt.plot(nu_prime, epsilon, label=f"z={z}")
plt.axvline(lyman_alpha_frequency, color="k", linestyle="--", label="Ly-alpha (z=20)")
plt.axvline(lyman_beta_frequency, color="r", linestyle="--", label="Ly-beta (z=20)")
plt.axvline(lyman_limit, color="g", linestyle="--", label="Lyman limit (z=20)")
#plt.loglog()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Total Epsilon_alpha^tot (arbitrary units)")
plt.legend(loc="upper right")
plt.savefig("epsilon_alpha_tot.png")
# plt.show()
