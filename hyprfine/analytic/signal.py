"""Functions to compute 21cm signal quantities."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import cosmology


@jax.jit
def T21(
    z: int,
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
    Omega_m = cosmo.Omega_b + cosmo.Omega_c
    return (
        54
        * (1 - xe)
        * ((1 - cosmo.Y_He) / 0.76)
        * (cosmo.Omega_b * (cosmo.H0 / 100) ** 2)
        / 0.02242
        * jnp.sqrt(
            0.1424 / (Omega_m * (cosmo.H0 / 100) ** 2) * ((1 + z) / 40)
        )
        * (1 - T_cmb / T_s)
    )  # 21cm brightness temperature in mKelvin
