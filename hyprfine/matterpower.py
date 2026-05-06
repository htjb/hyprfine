"""Calculate the matter power spectrum."""

import jax
import jax.numpy as jnp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

from hyprfine.parameters import cosmology

@jax.jit
def matterpowerspec(cosmo: cosmology, z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the matter power spectrum at redshift z.

    Args:
        cosmo: Cosmology parameters.
        z: Redshift.

    Returns:
        Matter power spectrum at redshift z.
    """
    cosmo_params = {
        "omega_b": jnp.array([cosmo.Omega_b * (cosmo.H0 / 100) ** 2]),
        "omega_cdm": jnp.array([cosmo.Omega_c * (cosmo.H0 / 100) ** 2]),
        "h": jnp.array([cosmo.H0 / 100]),
        "n_s": jnp.array([cosmo.ns]),
        "ln10^{10}A_s": jnp.array([cosmo.ln1010As]),
        "z": jnp.array([z]),
    }
    emulator = CPJ(probe="mpk_lin")
    emulator_predictions = emulator.predict(cosmo_params)
    return emulator.modes, emulator_predictions
