import jax.numpy as jnp

from hyprfine.analytic.sfrd import mean_sfrd
from hyprfine.utils.cosmology import n_H_tot


def nion_dot(z, cosmo, astro):
    mu_b_Msun = 1.22 * 1.67352e-27 / 1.989e30
    yr_to_s = 3.154e7
    Mpc_to_cm = 3.086e24

    Mh = 10 ** jnp.linspace(8, 13, 1000)
    sfrd = mean_sfrd(z, Mh, astro, cosmo)  # Msun/yr/ comoving Mpc^3

    # Convert comoving SFRD to physical ionizing photon rate density
    sfrd_cgs = sfrd / yr_to_s / Mpc_to_cm**3  # Msun/s/comoving cm^3
    sfrd_phys = sfrd_cgs * (1 + z) ** 3  # Msun/s/physical cm^3
    return astro.f_esc * astro.N_ion * sfrd_phys / mu_b_Msun  # photons/s/physical cm^3
