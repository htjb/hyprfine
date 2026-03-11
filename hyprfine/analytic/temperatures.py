"""Temperature calculations for the 21-cm signal."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import const


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


def Tc(T_gas: jnp.ndarray, T_s: jnp.ndarray) -> jnp.ndarray:
    """Calculate the effective colour temperature. From Munoz 2023.

    Args:
        T_gas: Kinetic temperature in Kelvin.
        T_s: Spin temperature in Kelvin.

    Returns:
        T_c: Effective colour temperature in Kelvin.
    """
    gcol = 0.4055
    T_c = (
        T_gas ** (-1) + gcol * T_gas ** (-1) * (T_s ** (-1) - T_gas ** (-1))
    ) ** (-1)
    return T_c


@jax.jit
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
