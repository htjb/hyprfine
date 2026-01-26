"""Functions to compute 21cm signal quantities."""

import jax
import jax.numpy as jnp

from hyperfine.utils.comsology import n_H_tot
from hyperfine.utils.parameters import const, cosmology


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

    xc = (const.Tstar * kappa(z, Tk, xe) * nH) / (
        const.A10 * const.Tcmb0 * (1 + z)
    )

    return xc


@jax.jit
def kappa(z: int, Tk: jnp.ndarray, xe: jnp.ndarray) -> jnp.ndarray:
    """Calculate the collisional coupling coefficient kappa.

    Args:
        z: Redshift.
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


@jax.jit
def Tcmb(z: int) -> jnp.ndarray:
    """Calculate CMB temperature at redshift z.

    Args:
        z: Redshift.

    Returns:
        T: CMB temperature in Kelvin.
    """
    T = const.Tcmb0 * (1 + z)  # CMB temperature in Kelvin, scaled by redshift
    return T


def Ts(T_gas: jnp.ndarray, T_cmb: jnp.ndarray, xc: jnp.ndarray) -> jnp.ndarray:
    """Calculate spin temperature.

    Args:
        T_gas: Kinetic temperature in Kelvin.
        T_cmb: CMB temperature in Kelvin.
        xc: Coupling coefficient.

    Returns:
        Tspin: Spin temperature in Kelvin.
    """
    Tspin = (T_cmb ** (-1) + xc * T_gas ** (-1)) / (1 + xc)
    return Tspin ** (-1)


@jax.jit
def T21(
    z: int,
    T_gas: jnp.ndarray,
    T_cmb: jnp.ndarray,
    T_s: jnp.ndarray,
    xe: jnp.ndarray,
    cosmo: cosmology,
) -> jnp.ndarray:
    """Calculate 21cm brightness temperature.

    Args:
        z: Redshift.
        T_gas: Kinetic temperature in Kelvin.
        T_cmb: CMB temperature in Kelvin.
        T_s: Spin temperature in Kelvin.
        xe: Free electron fraction.
        cosmo: Cosmology object.

    Returns:
        T21: 21cm brightness temperature in mKelvin.
    """
    return (
        54
        * (1 - xe)
        * ((1 - cosmo.Y_He) / 0.76)
        * (cosmo.Omega_bh2)
        / 0.02242
        * jnp.sqrt(
            0.1424 / (cosmo.Omega_m * (cosmo.H0 / 100) ** 2) * ((1 + z) / 40)
        )
        * (1 - T_cmb / T_s)
    )  # 21cm brightness temperature in mKelvin
