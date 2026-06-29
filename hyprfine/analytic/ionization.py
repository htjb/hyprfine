"""Ionization related functions."""

import jax
import jax.numpy as jnp

from hyprfine.analytic.sfrd import mean_sfrd
from hyprfine.parameters import astrophysics, conv, cosmology


@jax.jit
def nion_dot(z: float, cosmo: cosmology, astro: astrophysics) -> jnp.ndarray:
    """Calculate the ionizing photon production rate per unit volume at z.

    Args:
        z: Redshift.
        cosmo: Cosmology object.
        astro: Astrophysics object.

    Returns:
        nion_dot: Ionizing photon production rate per unit
            volume in photons/s/cm^3.
    """
    mu_b_Msun = 1.22 * 1.67352e-27 / 1.989e30

    Mh = 10 ** jnp.linspace(8, 13, 1000)
    sfrd = mean_sfrd(z, Mh, astro, cosmo)  # Msun/yr/ comoving Mpc^3

    # Convert comoving SFRD to physical ionizing photon rate density
    sfrd_cgs = sfrd / conv.yr_to_s / conv.Mpc_to_cm**3  # Msun/s/comoving cm^3
    sfrd_phys = sfrd_cgs * (1 + z) ** 3  # Msun/s/physical cm^3
    return (
        astro.f_esc * astro.N_ion * sfrd_phys / mu_b_Msun
    )  # photons/s/physical cm^3
