"""Test the generate signal function from hyprfine.analytic.main."""

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import astrophysics, cosmology


fgrid = jnp.linspace(5, 200, 100)  # Frequency grid in MHz
cosmo = cosmology(
    H0=67.36,
    Omega_b=0.049,
    Omega_c=0.266,
    Y_He=0.245,
    ns=0.97,
    ln1010As=3.044,
)  # Example cosmology parameters

astro = astrophysics(
    epsilon=0.1,
    alpha_star=0.5,
    beta_star=-0.5,
    M_pivot=3e11,
)

T21_values, xe, T_gas = generate_signal(
    f_grid=fgrid,
    cosmo=cosmo,
    astro=astro,
)


fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True)
signals = [T21_values, xe, T_gas]
labels = [r"$T_{21}$ [mK]", r"$x_e$", r"$T_k$ [K]"]
for sig, label, ax in zip(signals, labels, axes[:, 0]):
    ax.plot(fgrid, sig)
    if ax == axes[-1, 0]:
        ax.set_xlabel(r"$\nu$ [MHz]")
    ax.set_ylabel(label)
    ax.grid()


fgrid = jnp.linspace(5, 50, 100)  # Frequency grid in MHz
dT21dcosmo, dxedcosmo, dTgasdcosmo = jax.jacfwd(generate_signal, argnums=1)(
    fgrid, cosmo
)

signals = [dT21dcosmo, dxedcosmo, dTgasdcosmo]
labels = [
    [
        r"$\partial T_{21} / \partial \Omega_b$",
        r"$\partial x_e / \partial \Omega_b$",
        r"$\partial T_k / \partial \Omega_b$",
    ],
    [
        r"$\partial T_{21} / \partial H_0$",
        r"$\partial x_e / \partial H_0$",
        r"$\partial T_k / \partial H_0$",
    ],
]
for sig, label, ax in zip(signals, labels[0], axes[:, 1]):
    ax.plot(fgrid, sig.Omega_b)
    if ax == axes[-1, 1]:
        ax.set_xlabel(r"$\nu$ [MHz]")
    ax.set_ylabel(label)
    ax.grid()

for sig, label, ax in zip(signals, labels[1], axes[:, 2]):
    ax.plot(fgrid, sig.H0)
    if ax == axes[-1, 2]:
        ax.set_xlabel(r"$\nu$ [MHz]")
    ax.set_ylabel(label)
    ax.grid()

plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.savefig("grad_T21_vs_frequency.png")
plt.show()
