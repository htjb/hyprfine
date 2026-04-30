"""Test the Wouthuysen-Field coupling calculations."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.main import generate_signal
from hyprfine.analytic.coupling_coeffs import x_alpha
from hyprfine.analytic.wouthuysen_field import (
    J_alpha,
    calculate_epsilon_alpha_tot,
)
from hyprfine.parameters import astrophysics, cosmology

lyman_alpha_frequency = 2.466e15  # Hz
lyman_beta_frequency = 2.922e15  # Hz
lyman_limit = 3.289e15  # Hz


zs = [10, 12, 15, 20, 25, 30]

nu = 10 ** jnp.arange(
    jnp.log10(lyman_alpha_frequency), jnp.log10(lyman_limit), 0.001
)

# for z in zs:
#     epsilon = calculate_epsilon_alpha_tot(z_source=z, z_21=20)
#     nu_prime = nu * (1 + 20) / (1 + z)
#     plt.plot(nu_prime, epsilon, label=f"z={z}")
# plt.axvline(lyman_alpha_frequency, color="k", linestyle="--", label="Ly-alpha (z=20)")
# plt.axvline(lyman_beta_frequency, color="r", linestyle="--", label="Ly-beta (z=20)")
# plt.axvline(lyman_limit, color="g", linestyle="--", label="Lyman limit (z=20)")
# #plt.loglog()
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Total Epsilon_alpha^tot (arbitrary units)")
# plt.legend(loc="upper right")
# plt.savefig("epsilon_alpha_tot.png")
# #plt.show()
# plt.close()


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

xalpha = []
for z in zs:
    nu, jalpha_values = J_alpha(
        z=z,
        cosmo=cosmo,
        astro=astro,
        Mmin=1e8,
        Mmax=1e13,
        N_shells=200,
        z_max_source=50.0,
    )
    if z == 20:
        plt.plot(nu, jalpha_values)
        plt.xlabel("Frequency bin")
        plt.ylabel("J_alpha (arbitrary units)")
        plt.title("Lyman-alpha flux J_alpha at z=20")
        plt.grid()
        plt.savefig("J_alpha_z20.png")
        plt.close()

    xalpha.append(x_alpha(z=z, cosmo=cosmo, astro=astro, T_cmb0=2.725))

plt.plot(zs, xalpha, marker="o")
plt.xlabel("Redshift z")
plt.ylabel("Lyman-alpha coupling coefficient x_alpha")
plt.title("Lyman-alpha coupling coefficient x_alpha vs redshift")
plt.grid()
plt.savefig("x_alpha_vs_z.png")
#plt.show()
plt.close()

T21_values, xe, T_gas, xc_values, T_s, T_cmb, xalpha_values = generate_signal(
    f_grid=jnp.linspace(5, 200, 100),  # Frequency grid in MHz
    cosmo=cosmo,
    astro=astro,
    z_init=1100,
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
