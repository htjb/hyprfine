"""Coupling coefficients for the 21cm signal."""

import jax
from jax import numpy as jnp

from hyprfine.analytic.wouthuysen_field import J_alpha
from hyprfine.parameters import astrophysics, const, cosmology
from hyprfine.utils.cosmology import n_H_tot


@jax.jit
def x_alpha(
    z: float, cosmo: cosmology, astro: astrophysics, T_cmb0: float = 2.725
) -> jnp.ndarray:
    """Dimensionless Lyman-alpha coupling coefficient.

    Args:
        z: Redshift.
        cosmo: Cosmology parameters.
        astro: Astrophysics parameters.
        T_cmb0: CMB temperature at z=0 in Kelvin.

    Returns:
        x_alpha: Lyman-alpha coupling coefficient.
    """
    J_alpha_c_inv = 1.811e11 / (1 + z) * (2.725 / T_cmb0)  # cm2 s Hz sr
    # S_alpha is an order-unity correction factor, ~1 for now
    S_alpha = 1.0

    nu, jalpha_values = J_alpha(z, cosmo, astro)  # Get J_alpha values
    jalpha_alpha = jnp.interp(const.lyman_alpha_freq, nu, jalpha_values)
    return S_alpha * jalpha_alpha * J_alpha_c_inv


@jax.jit
def xc(
    z: int, xe: jnp.ndarray, Tk: jnp.ndarray, cosmo: cosmology
) -> jnp.ndarray:
    """Calculate the coupling coefficient xc.

    Args:
        z: Redshift.
        xe: Free electron fraction.
        Tk: Kinetic temperature in Kelvin.
        cosmo: Cosmology object.

    Returns:
        xc: Coupling coefficient.
    """
    nH = n_H_tot(z, cosmo)

    xc = (const.Tstar * kappa(Tk, xe) * nH) / (
        const.A10 * const.Tcmb0 * (1 + z)
    )

    return xc


@jax.jit
def kappa(Tk: jnp.ndarray, xe: jnp.ndarray) -> jnp.ndarray:
    """Calculate the collisional coupling coefficient kappa.

    Args:
        Tk: Kinetic temperature in Kelvin.
        xe: Free electron fraction.

    Returns:
        kappa: Collisional coupling coefficient in m^3/s.
    """
    # H-H collisions (Zygelman 2005, valid up to ~300K)
    kappa_HH = 3.1e-11 * Tk**0.357 * jnp.exp(-32.0 / Tk) * 1e-6  # m^3/s

    # e-H collisions from https://arxiv.org/pdf/2108.00115
    kappa_eH = (
        10
        ** (
            -9.607
            + 0.5 * jnp.log10(Tk) * jnp.exp(-((jnp.log10(Tk)) ** 4) / 1800.0)
        )
        * 1e-6
    )

    # p-H collisions (assume ~same as e-H to leading order)
    # there is a factor related to the ratio of masses but
    # they are subdominant anyway
    kappa_pH = kappa_eH

    return kappa_HH * (1 - xe) + (kappa_eH + kappa_pH) * xe
