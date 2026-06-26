"""Benchmark hyprfine signal generation on CPU (and CUDA GPU if available)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zeus21

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import cosmology, astrophysics

skip_gpu = False
if len(sys.argv) > 1 and sys.argv[1] == "--no-gpu":
    skip_gpu = True

# ---------------------------------------------------------------------------
# Shared fiducial parameters (Planck 2018 cosmology)
# ---------------------------------------------------------------------------
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
alpha_low = 0.14
alpha_high = -8.0
N_alpha = 9690
f_esc = 0.1
N_ion = 5000.0

planck = cosmology(
    H0=H0, Omega_b=Omega_b, Omega_c=Omega_c, Y_He=Y_He,
    ns=ns, ln1010As=ln1010As,
)
astro = astrophysics(
    epsilon=epsilon, alpha_star=alpha_star, beta_star=beta_star,
    M_pivot=M_pivot, L40=L40, alpha_x=alpha_x, nu_0=E0_keV,
    alpha_low=alpha_low, alpha_high=alpha_high, N_alpha=N_alpha,
    f_esc=f_esc, N_ion=N_ion,
)

f_grid = jnp.linspace(5.0, 250.0, 500)  # MHz


def time_device(device: jax.Device) -> tuple[float, float]:
    """Time cold and warm signal generation on a given JAX device.

    Args:
        device: JAX device to run on.

    Returns:
        Tuple of (cold_time, warm_time) in seconds.
    """
    f = jax.device_put(f_grid, device)

    # Cold run — includes JIT compilation
    t0 = time.perf_counter()
    sig, xe, T_gas = generate_signal(f, planck, astro)
    jax.block_until_ready(sig)
    cold = time.perf_counter() - t0

    # Warm run
    t0 = time.perf_counter()
    sig, xe, T_gas = generate_signal(f, planck, astro)
    jax.block_until_ready(sig)
    warm = time.perf_counter() - t0

    return cold, warm, sig


# ---------------------------------------------------------------------------
# Always benchmark CPU
# ---------------------------------------------------------------------------
cpu = jax.devices("cpu")[0]
print("Benchmarking CPU...")
cpu_cold, cpu_warm, signal = time_device(cpu)
print(f"  Cold: {cpu_cold:.3f} s   Warm: {cpu_warm:.3f} s")

results = {"CPU (cold)": cpu_cold, "CPU (warm)": cpu_warm}

# ---------------------------------------------------------------------------
# Optionally benchmark CUDA GPU
# ---------------------------------------------------------------------------
try:
    if not skip_gpu:
        gpus = jax.devices("gpu")
        if gpus:
            gpu = gpus[0]
            print(f"Benchmarking GPU ({gpu.device_kind})...")
            gpu_cold, gpu_warm, _ = time_device(gpu)
            print(f"  Cold: {gpu_cold:.3f} s   Warm: {gpu_warm:.3f} s")
            results["GPU (cold)"] = gpu_cold
            results["GPU (warm)"] = gpu_warm
except RuntimeError:
    gpus = None
    print("No CUDA GPU found — skipping GPU benchmark.")

# ---------------------------------------------------------------------------
# zeus21 CPU benchmark
# ---------------------------------------------------------------------------
user_params = zeus21.User_Parameters()
cosmo_input = zeus21.Cosmo_Parameters_Input(
    omegab=omegab, omegac=omegac, h_fid=h, As=As, ns=ns,
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

print("Benchmarking zeus21 (CPU)...")

t0 = time.perf_counter()
T21c = zeus21.get_T21_coefficients(
    user_params, CosmoParams, ClassyCosmo, astro_params, HMFintclass,
    zmin=10.0,
)
z21_time = time.perf_counter() - t0
print(f"  {z21_time:.3f} s")

results["zeus21"] = z21_time

# ---------------------------------------------------------------------------
# Plot: signal + bar chart
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax_sig = fig.add_subplot(gs[0, :2])
ax_bar = fig.add_subplot(gs[0, 2])

# Signal
z_grid = 1420.4 / np.array(f_grid) - 1
print("z_grid:", z_grid.min(), "-", z_grid.max())
ax_sig.plot(
    z_grid, np.array(signal), color="steelblue", lw=1.5, label="hyprfine",
)
ax_sig.plot(
    T21c.zintegral, T21c.T21avg,
    color="seagreen", lw=1.5, ls="--", label="zeus21",
)
ax_sig.set_xlabel("Redshift $z$")
ax_sig.set_ylabel(r"$T_{21}$ [mK]")
ax_sig.set_title("Dark ages 21-cm signal (Planck 2018)")
ax_sig.set_xscale("log")
ax_sig.legend(fontsize=8)

# Bar chart
labels = list(results.keys())
print(labels)
times = list(results.values())
colours = [
    "seagreen" if "zeus21" in ll
    else "steelblue" if "CPU" in ll
    else "darkorange"
    for ll in labels
]
hatches = [
    "" if ("warm" in ll.lower() or "zeus21" in ll) else "///"
    for ll in labels
]
bars = ax_bar.bar(
    labels, times, color=colours, hatch=hatches, edgecolor="white"
)
for bar, t in zip(bars, times):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.01,
        f"{t:.3f} s",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax_bar.set_ylabel("Wall time [s]")
if gpus:
    ax_bar.set_title(
        "Signal generation benchmark\n" + "(" + gpu.device_kind + ")"
    )
else:
    ax_bar.set_title("Signal generation benchmark")
ax_bar.tick_params(axis="x", rotation=15)

plt.savefig("bin/benchmark.png", dpi=150)
print("Saved benchmark.png")
