import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config

from hyprfine.analytic.coupling_coeffs import xc
from hyprfine.analytic.main import generate_signal
from hyprfine.analytic.temperatures import Tcmb, Ts
from hyprfine.parameters import cosmology
from hyprfine.recombination.emulator import call_hyrec_emulator

config.update("jax_enable_x64", True)


@jax.jit
def kappa(
    Tk: jnp.ndarray, xe: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate the collisional coupling coefficient kappa.

    From https://arxiv.org/pdf/2108.00115.

    Args:
        Tk: Kinetic temperature in Kelvin.
        xe: Free electron fraction.

    Returns:
        kappa: Collisional coupling coefficient in m^3/s.
    """
    # H-H collisions
    kappa_HH = 3.1e-11 * Tk**0.357 * jnp.exp(-32.0 / Tk) * 1e-6  # m^3/s

    # e-H collisions
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

    return (
        kappa_HH * (1 - xe) + (kappa_eH + kappa_pH) * xe,
        kappa_HH * (1 - xe),
        kappa_eH * xe,
        kappa_pH * xe,
    )


cosmo = cosmology()

fgrid = jnp.linspace(3, 60, 100)  # MHz
zgrid = 1420.4 / fgrid - 1

xe, Tk = call_hyrec_emulator(
    z_grid=zgrid,
    H0=cosmo.H0,
    omb=cosmo.Omega_b,
    omc=cosmo.Omega_c,
    yhe=cosmo.Y_He,
)

vmappedxc = jax.vmap(xc, in_axes=(0, 0, 0, None))
xc_values = vmappedxc(zgrid, xe, Tk, cosmo)

kappa, kappa_HH, kappa_eH, kappa_pH = jax.vmap(kappa, in_axes=(0, 0))(Tk, xe)

def f_to_z(f):
    return 1420.4 / f - 1


def z_to_f(z):
    return 1420.4 / (z + 1)


def add_z_axis(ax):
    secax = ax.secondary_xaxis("top", functions=(f_to_z, z_to_f))
    secax.set_ticks([int(round(f_to_z(f))) for f in range(10, 60, 10)])
    secax.set_xlabel(r"$z$")


fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes = axes.flatten()

axes[0].plot(fgrid, xc_values)
axes[0].set_xlabel(r"$\nu$ [MHz]")
axes[0].set_ylabel(r"$x_c$")
add_z_axis(axes[0])

axes[1].plot(fgrid, kappa, label=r"Total $\kappa$")
axes[1].plot(fgrid, kappa_HH, label=r"$\kappa_{HH} ( 1 - x_e )$")
axes[1].plot(fgrid, kappa_eH, label=r"$\kappa_{eH} x_e$")
axes[1].plot(fgrid, kappa_pH, label=r"$\kappa_{pH} x_e$")
axes[1].set_xlabel(r"$\nu$ [MHz]")
axes[1].set_ylabel(r"$\kappa$ [m$^3$/s]")
axes[1].legend()
axes[1].set_yscale("log")
add_z_axis(axes[1])

T21, xe, Tk = generate_signal(fgrid, cosmo)
Tcmb_values = Tcmb(zgrid)
Ts_values = Ts(Tk, Tcmb_values, xc_values, jnp.zeros_like(xc_values))

axes[2].plot(fgrid, Tcmb_values, label=r"$T_{\mathrm{CMB}}$")
axes[2].plot(fgrid, Tk, label=r"$T_k$")
axes[2].plot(fgrid, Ts_values, label=r"$T_s$")
axes[2].set_xlabel(r"$\nu$ [MHz]")
axes[2].set_ylabel(r"Temperature (K)")
axes[2].legend()
axes[2].loglog()
add_z_axis(axes[2])

axes[3].plot(fgrid, T21)
axes[3].set_xlabel(r"$\nu$ [MHz]")
axes[3].set_ylabel(r"$T_{21}$ [mK]")
add_z_axis(axes[3])

plt.tight_layout()
plt.savefig("docs/figures/dark-ages.png", dpi=300)
plt.show()
