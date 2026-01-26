"""Cosmological helpers for hyperfine."""

import jax
import jax.numpy as jnp

from hyprfine.utils.parameters import const, cosmology


@jax.jit
def n_H_tot(z: int, cosmo: cosmology) -> jnp.ndarray:
    """Mean hydrogen number density in m^-3.

    Args:
        z: Redshift.
        cosmo: Cosmology object.

    Returns:
        n_H: Mean hydrogen number density in m^-3.
    """
    H0 = cosmo.H0 * 1e3 / 3.086e22
    rho_bar = (
        (3 * H0**2 / (8 * jnp.pi * const.G)) * cosmo.Omega_b * (1 + z) ** 3
    )
    return ((1 - cosmo.Y_He) * rho_bar) / const.m_p
