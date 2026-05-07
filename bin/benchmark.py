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

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import cosmology, astrophysics

skip_gpu = False
if len(sys.argv) > 1 and sys.argv[1] == "--no-gpu":
    skip_gpu = True

# ---------------------------------------------------------------------------
# Planck 2018 cosmology
# ---------------------------------------------------------------------------
planck = cosmology()

astro = astrophysics()

f_grid = jnp.linspace(5.0, 300.0, 500)  # MHz


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
    print("No CUDA GPU found — skipping GPU benchmark.")

# ---------------------------------------------------------------------------
# Plot: signal + bar chart
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax_sig = fig.add_subplot(gs[0, :2])
ax_bar = fig.add_subplot(gs[0, 2])

# Signal
z_grid = 1420.4 / np.array(f_grid) - 1
ax_sig.plot(z_grid, np.array(signal), color="steelblue", lw=1.5)
ax_sig.set_xlabel("Redshift $z$")
ax_sig.set_ylabel(r"$T_{21}$ [mK]")
ax_sig.set_title("Dark ages 21-cm signal (Planck 2018)")
ax_sig.set_xscale("log")

# Bar chart
labels = list(results.keys())
times = list(results.values())
colours = ["steelblue" if "CPU" in ll else "darkorange" for ll in labels]
hatches = ["" if "warm" in ll.lower() else "///" for ll in labels]
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
ax_bar.set_title("Signal generation benchmark")
ax_bar.tick_params(axis="x", rotation=15)

plt.savefig("bin/benchmark.png", dpi=150)
print("Saved benchmark.png")
