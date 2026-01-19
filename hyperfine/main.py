"""Functions to compute the 21cm signal from the Dark Ages."""

import jax
import jax.numpy as jnp

from hyperfine.recombination.hyrec import call_hyrec, set_up_hyrec
from hyperfine.signal.signal import T21, Tcmb, Ts, xc
from hyperfine.utils.parameters import cosmology

xcvmap = jax.vmap(xc, in_axes=(0, 0, 0, None))
vmappedT21 = jax.vmap(T21, in_axes=(0, 0, 0, 0, 0, None))


def generate_signal(
    f_grid: jnp.ndarray,
    sample: jnp.ndarray,
    z_init: int,
    detailed_output: bool = False,
    verbose: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate 21cm signal for a given cosmological sample.

    Args:
        f_grid: Frequency grid in MHz.
        sample: Cosmological parameters [H0, Omega_m, Omega_b, Omega_c,
                Y_He].
        z_init: Initial redshift.
        detailed_output: Whether to return detailed outputs (xe, Tk, xc).
        verbose: Whether to print verbose output from the recombination codes.

    Returns:
        T21_values: 21cm brightness temperature values over the frequency grid.
    """
    cosmo = cosmology(
        H0=sample[0],
        Omega_m=sample[1],
        Omega_b=sample[2] / (sample[0] / 100) ** 2,
        Omega_c=sample[3] / (sample[0] / 100) ** 2,
        Omega_bh2=sample[2],
        Omega_ch2=sample[3],
        z_init=z_init,
        Y_He=sample[4],
    )  # Example cosmology parameters
    try:
        set_up_hyrec(
            H0=cosmo.H0,
            omb=cosmo.Omega_b,
            omc=cosmo.Omega_c,
            omk=0.0,
            yhe=cosmo.Y_He,
            base_dir="./",
        )
        z_grid = 1420.4 / (f_grid) - 1
        xe, T_gas = call_hyrec(
            base_dir="./", redshift=z_grid, verbose=verbose
        )
        xe, T_gas = jnp.array(xe), jnp.array(T_gas)

        xc_values = xcvmap(z_grid, xe, T_gas, cosmo)

        T_cmb = Tcmb(z_grid)

        T_s = Ts(T_gas, T_cmb, xc_values)

        T21_values = vmappedT21(z_grid, T_gas, T_cmb, T_s, xe, cosmo)
        if detailed_output:
            return T21_values, xe, T_gas, xc_values
        else:
            return T21_values
    except Exception as e:
        print(f"Error generating signal for sample {sample}: {e}")
        return jnp.full_like(f_grid, jnp.nan)
