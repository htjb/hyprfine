from jax import config
config.update("jax_enable_x64", True)

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.parameters import cosmology, astrophysics
from hyprfine.recombination.odes import evolve_igm
from hyprfine.recombination.emulator import call_hyrec_emulator



f_grid = jax.numpy.linspace(5, 200, 500)  # Example frequency grid in MHz

cosmo = cosmology(
    H0=67.36,
    Omega_m=0.315,
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

z_grid = 1420.4 / (f_grid) - 1

xe, T_gas = call_hyrec_emulator(
    z_grid=z_grid,
    H0=cosmo.H0,
    omb=cosmo.Omega_b,
    omc=cosmo.Omega_c,
    yhe=cosmo.Y_He,
)

xe_z50 = jnp.interp(50, z_grid[::-1], xe[::-1])
T_gas_z50 = jnp.interp(50, z_grid[::-1], T_gas[::-1])

z_out, evolved_Tk, evolved_xe = evolve_igm(
    z_start=50,
    z_end=z_grid[-1],
    Tk_init=T_gas_z50,
    xe_init=xe_z50,
    cosmo=cosmo,
    astro=astro,
)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(z_out, evolved_xe, label="Evolved xe")
axes[0].plot(z_grid, xe, label="Emulator xe", linestyle="dashed")
axes[0].set_xlabel("Redshift z")
axes[0].set_ylabel("Ionization fraction xe")
axes[0].set_title("Ionization fraction xe vs redshift")
axes[0].legend()
axes[0].grid()
axes[1].plot(z_out, evolved_Tk, label="Evolved T_k")
axes[1].plot(z_grid, T_gas, label="Emulator T_k", linestyle="dashed")
axes[1].set_xlabel("Redshift z")
axes[1].set_ylabel("Kinetic temperature T_k (K)")
axes[1].set_title("Kinetic temperature T_k vs redshift")
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.savefig("xe_Tk_evolution.png")
# plt.show()
plt.close()