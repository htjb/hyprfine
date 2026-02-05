"""Functions to compute 21cm signal quantities."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import const, cosmology


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
        * (cosmo.Omega_b * (cosmo.H0 / 100) ** 2)
        / 0.02242
        * jnp.sqrt(
            0.1424 / (cosmo.Omega_m * (cosmo.H0 / 100) ** 2) * ((1 + z) / 40)
        )
        * (1 - T_cmb / T_s)
    )  # 21cm brightness temperature in mKelvin
