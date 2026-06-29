"""Functions to compute the 21cm signal from the Dark Ages."""

import jax
import jax.numpy as jnp

from hyprfine.analytic.coupling_coeffs import x_alpha, xc
from hyprfine.analytic.signal import T21
from hyprfine.analytic.temperatures import Tcmb, Ts
from hyprfine.parameters import astrophysics, cosmology
from hyprfine.recombination.emulator import call_hyrec_emulator
from hyprfine.recombination.odes import evolve_igm

vmappedxc = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, None))
vmappedxalpha = jax.vmap(x_alpha, in_axes=(0, None, None, None))
vmappedTs = jax.vmap(Ts, in_axes=(0, 0, 0, 0))
vmappedTcmb = jax.vmap(Tcmb, in_axes=(0,))

@jax.jit
def generate_signal(
    f_grid: jnp.ndarray,
    cosmo: cosmology,
    astro: astrophysics | None = None,
) -> (
    tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
):
    """Generate 21cm signal for a given cosmological sample.

    Args:
        f_grid: Frequency grid in MHz.
        cosmo: Cosmological parameters [H0, Omega_b, Omega_c,
                Y_He].
        astro: Astrophysical parameters.
        skip_cosmic_dawn: Whether to skip cosmic dawn.
        detailed_output: Whether to return detailed outputs (xe, Tk, xc).

    Returns:
        T21_values: 21cm brightness temperature values over the frequency grid.
    """
    z_grid = 1420.4 / (f_grid) - 1
    xe, T_gas = call_hyrec_emulator(
        z_grid=z_grid,
        H0=cosmo.H0,
        omb=cosmo.Omega_b,
        omc=cosmo.Omega_c,
        yhe=cosmo.Y_He,
    )

    if astro is not None:
        xalpha_values = vmappedxalpha(z_grid, cosmo, astro, Tcmb(0))
        T_gas_z50 = jnp.interp(35, z_grid[::-1], T_gas[::-1])
        xe_z50 = jnp.interp(35, z_grid[::-1], xe[::-1])

        z_out, evolved_Tk, evolved_xe = evolve_igm(
            z_start=35,
            z_end=z_grid[-1],
            Tk_init=T_gas_z50,
            xe_init=xe_z50,
            cosmo=cosmo,
            astro=astro,
        )

        # Compute evolved values over the full grid
        T_gas_evolved = jnp.interp(z_grid, z_out[::-1], evolved_Tk[::-1])
        xe_evolved = jnp.interp(z_grid, z_out[::-1], evolved_xe[::-1])
        # Use hyrec above z=35, evolved below z=35
        T_gas = jnp.where(z_grid >= 35, T_gas, T_gas_evolved)
        xe = jnp.where(z_grid >= 35, xe, xe_evolved)
    else:
        xalpha_values = jnp.zeros_like(z_grid)

    xc_values = vmappedxc(z_grid, xe, T_gas, cosmo)

    T_cmb = vmappedTcmb(z_grid)

    T_s = vmappedTs(T_gas, T_cmb, xc_values, xalpha_values)

    T21_values = vmappedT21(z_grid, T_cmb, T_s, xe, cosmo)
    return T21_values, xe, T_gas
