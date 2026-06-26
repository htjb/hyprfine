"""Compare the hyprfine and zeus21 global 21-cm signals.

Runs both codes with identical parameters and plots T21, Tk, and xe with
fractional residual panels.

Run with:
    python bin/compare_signal.py
"""

from jax import config

config.update("jax_enable_x64", True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import zeus21

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import astrophysics, cosmology

# ── Shared fiducial parameters ───────────────────────────────────────────────

h = 0.6781
H0 = h * 100.0
omegab = 0.0223828
omegac = 0.1201075
Omega_b = omegab / h**2
Omega_c = omegac / h**2
Y_He = 0.245
ns = 0.9660499
As = 2.100549e-09
ln1010As = float(np.log(As * 1e10))

epsilon = 0.1
alpha_star = 0.5
beta_star = -0.5
M_pivot = 3e11
L40 = 3.0
alpha_x = -1.0
E0_keV = 0.5
N_alpha = 9690
alpha_low = 0.14
alpha_high = -8.0
f_esc = 0.1
N_ion = 5000.0

# ── zeus21 ───────────────────────────────────────────────────────────────────

print("Running zeus21 …")
user_params = zeus21.User_Parameters()
cosmo_input = zeus21.Cosmo_Parameters_Input(
    omegab=omegab,
    omegac=omegac,
    h_fid=h,
    As=As,
    ns=ns,
)
CosmoParams, ClassyCosmo, CorrFClass, HMFintclass = zeus21.cosmo_wrapper(
    user_params, cosmo_input
)
astro_params = zeus21.Astro_Parameters(
    user_params,
    CosmoParams,
    alphastar=alpha_star,
    betastar=beta_star,
    epsstar=epsilon,
    Mc=M_pivot,
    L40_xray=L40,
    E0_xray=E0_keV * 1000,
    alpha_xray=alpha_x,
    Nalpha_lyA_II=N_alpha,
    USE_POPIII=False,
    USE_LW_FEEDBACK=False,
)
T21c = zeus21.get_T21_coefficients(
    user_params,
    CosmoParams,
    ClassyCosmo,
    astro_params,
    HMFintclass,
    zmin=10.0,
)
z_z21 = T21c.zintegral
T21_z21 = T21c.T21avg
Tk_z21 = T21c.Tk_avg
xe_z21 = T21c.xe_avg
print(f"  done — z = {z_z21.min():.1f}–{z_z21.max():.1f}")

# ── hyprfine ─────────────────────────────────────────────────────────────────

print("Running hyprfine …")
cosmo = cosmology(
    H0=H0,
    Omega_b=Omega_b,
    Omega_c=Omega_c,
    Y_He=Y_He,
    ns=ns,
    ln1010As=ln1010As,
)
astro = astrophysics(
    epsilon=epsilon,
    alpha_star=alpha_star,
    beta_star=beta_star,
    M_pivot=M_pivot,
    L40=L40,
    alpha_x=alpha_x,
    nu_0=E0_keV,
    alpha_low=alpha_low,
    alpha_high=alpha_high,
    N_alpha=N_alpha,
    f_esc=f_esc,
    N_ion=N_ion,
)

f_grid = jnp.array(1420.4 / (z_z21[::-1] + 1))
T21_hypr, xe_hypr, Tk_hypr = generate_signal(f_grid, cosmo, astro)

T21_hypr = np.array(T21_hypr)[::-1]
xe_hypr = np.array(xe_hypr)[::-1]
Tk_hypr = np.array(Tk_hypr)[::-1]
print("  done.")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(12, 8))
outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
gs = [
    gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[i],
        height_ratios=[3, 1],
        hspace=0.07,
    )
    for i in range(3)
]

kw_z21 = dict(color="tab:blue", lw=2, label="zeus21")
kw_hypr = dict(color="tab:orange", lw=2, ls="--", label="hyprfine")
kw_res = dict(color="tab:green", lw=1.5)


def style_res(ax, ylabel):
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlabel("Redshift $z$")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)


# ── T21 ──────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0][0])
axr = fig.add_subplot(gs[0][1], sharex=ax)
ax.plot(z_z21, T21_z21, **kw_z21)
ax.plot(z_z21, T21_hypr, **kw_hypr)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_ylabel(r"$\overline{T}_{21}$ [mK]")
ax.set_title("Global 21-cm signal")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), visible=False)
axr.plot(z_z21, T21_hypr - T21_z21, **kw_res)
style_res(axr, "resid.\n[mK]")

# ── Tk ───────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1][0])
axr = fig.add_subplot(gs[1][1], sharex=ax)
ax.semilogy(z_z21, Tk_z21, **kw_z21)
ax.semilogy(z_z21, Tk_hypr, **kw_hypr)
ax.set_ylabel(r"$T_k$ [K]")
ax.set_title("Gas kinetic temperature")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), visible=False)
axr.plot(z_z21, Tk_hypr / Tk_z21 - 1, **kw_res)
style_res(axr, "frac.\nresid.")

# ── xe ───────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2][0])
axr = fig.add_subplot(gs[2][1], sharex=ax)
ax.semilogy(z_z21, xe_z21, **kw_z21)
ax.semilogy(z_z21, xe_hypr, **kw_hypr)
ax.set_ylabel(r"$x_e$")
ax.set_title("Free electron fraction")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), visible=False)
axr.plot(z_z21, xe_hypr / xe_z21 - 1, **kw_res)
style_res(axr, "frac.\nresid.")

fig.suptitle(
    "hyprfine vs zeus21 — Planck 2018, fiducial astrophysics\n"
    r"$\epsilon_*=0.1,\ L_{40}=3,\ \alpha_X=-1,\ f_{\rm esc}=0.1$",
    fontsize=10,
)

out = "compare_signal.png"
plt.savefig('bin/' + out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
