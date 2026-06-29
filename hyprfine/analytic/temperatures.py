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


@jax.jit
def Ts(
    T_gas: jnp.ndarray,
    T_cmb: jnp.ndarray,
    xc: jnp.ndarray,
    xalpha: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate spin temperature.

    Args:
        T_gas: Kinetic temperature in Kelvin.
        T_cmb: CMB temperature in Kelvin.
        xc: Coupling coefficient.
        xalpha: Lyman-alpha coupling coefficient.

    Returns:
        Tspin: Spin temperature in Kelvin.
    """
    Tspin = (T_cmb ** (-1) + xc * T_gas ** (-1) + xalpha * T_gas ** (-1)) / (
        1 + xc + xalpha
    )
    return Tspin ** (-1)
