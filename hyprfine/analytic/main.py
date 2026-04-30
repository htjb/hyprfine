"""Functions to compute the 21cm signal from the Dark Ages."""

import jax
import jax.numpy as jnp

from hyprfine.analytic.coupling_coeffs import xc, x_alpha
from hyprfine.analytic.signal import T21
from hyprfine.analytic.temperatures import Tcmb, Ts
from hyprfine.parameters import cosmology, astrophysics
from hyprfine.recombination.emulator import call_hyrec_emulator

xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))
vmappedxalpha = jax.vmap(x_alpha, in_axes=(0, None, None, None))


def generate_signal(
    f_grid: jnp.ndarray,
    cosmo: cosmology,
    astro: astrophysics | None = None,
    detailed_output: bool = False,
) -> (
    jnp.ndarray
    | tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
):
    """Generate 21cm signal for a given cosmological sample.

    Args:
        f_grid: Frequency grid in MHz.
        cosmo: Cosmological parameters [H0, Omega_m, Omega_b, Omega_c,
                Y_He].
        astro: Astrophysical parameters.
        skip_cosmic_dawn: Whether to skip cosmic dawn.
        detailed_output: Whether to return detailed outputs (xe, Tk, xc).

    Returns:
        T21_values: 21cm brightness temperature values over the frequency grid.
    """
    try:
        z_grid = 1420.4 / (f_grid) - 1
        xe, T_gas = call_hyrec_emulator(
            z_grid=z_grid,
            H0=cosmo.H0,
            omb=cosmo.Omega_b,
            omc=cosmo.Omega_c,
            yhe=cosmo.Y_He,
        )
        if astro is None:
            xalpha_values = jnp.zeros_like(z_grid)
        else:
            xalpha_values = vmappedxalpha(
                z_grid, cosmo, astro, Tcmb(0)
            )
        # use hyrec down to z=50 then solve my own ODE for Tk with
        # X-ray heating term and my own ODE for xe with
        # sed of ionizing sources. same process for Xray and ionizing flux as
        # for J_alpha in wouthuysen_field.py, but with different seds
        T_gas_z50 = jnp.interp(50, z_grid[::-1], T_gas[::-1])
        T_gas_beyond_z50 = (
            T_gas_z50 * (1 + z_grid[z_grid <= 50]) ** 2 / (1 + 50) ** 2
        )
        T_gas = jnp.concat([T_gas[z_grid >= 50], T_gas_beyond_z50])

        # xe_z50 = jnp.interp(50, z_grid[::-1], xe[::-1])
        xe_beyond_z50 = jnp.ones_like(z_grid[z_grid <= 10])
        xe = jnp.concat([xe[z_grid >= 10], xe_beyond_z50])

        xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

        T_cmb = Tcmb(z_grid)

        T_s = Ts(T_gas, T_cmb, xc_values, xalpha_values)

        T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
        if detailed_output:
            return T21_values, xe, T_gas, xc_values, T_s, T_cmb, xalpha_values
        else:
            return T21_values
    except Exception as e:
        print(f"Error generating signal for sample {cosmo}: {e}")
        return jnp.full_like(f_grid, jnp.nan)
