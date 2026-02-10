"""Cosmological helpers for hyperfine."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import const, cosmology


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


def rhom(z: int, cosmo: cosmology) -> jnp.ndarray:
    """Mean matter density in M_sun/Mpc^3.

    Args:
        z: Redshift.
        cosmo: Cosmology object.

    Returns:
        rhom: Mean matter density in M_sun/Mpc^3.
    """
    H0 = cosmo.H0 * 1e3 / const.Mpc

    rho_bar = (
        (3 * H0**2 / (8 * jnp.pi * const.G))
        * (cosmo.Omega_b + cosmo.Omega_c)
        * (1 + z) ** 3
    )

    conversion_factor = const.Mpc**3 / const.Msun

    return rho_bar * conversion_factor


def growth_factor(z: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Linear growth factor D(z), normalized to D(0)=1.

    Valid for flat LCDM.

    Args:
        z: Redshift.
        cosmo: cosmology parameters.

    Returns:
        D(z): Linear growth factor at redshift z.
    """
    Omega_m0 = cosmo.Omega_b + cosmo.Omega_c
    Omega_L0 = 1.0 - Omega_m0

    def E2(z: jnp.ndarray) -> jnp.ndarray:
        return Omega_m0 * (1 + z) ** 3 + Omega_L0

    def Omega_m(z: jnp.ndarray) -> jnp.ndarray:
        return Omega_m0 * (1 + z) ** 3 / E2(z)

    def Omega_L(z: jnp.ndarray) -> jnp.ndarray:
        return Omega_L0 / E2(z)

    def g(z: jnp.ndarray) -> jnp.ndarray:
        Om = Omega_m(z)
        Ol = Omega_L(z)
        return (5 * Om / 2) / (
            Om ** (4 / 7) - Ol + (1 + Om / 2) * (1 + Ol / 70)
        )

    return g(z) / (g(0.0) * (1 + z))
