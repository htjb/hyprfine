"""Benchmark hyprfine signal generation: wall time vs batch size."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hyprfine.analytic.main import generate_signal  # noqa: E402
from hyprfine.parameters import astrophysics, cosmology  # noqa: E402

skip_gpu = "--no-gpu" in sys.argv

planck = cosmology()
astro = astrophysics()
f_grid = jnp.linspace(5.0, 250.0, 500)

BATCH_SIZES = [1, 10, 100, 1000]
N_REPEATS = 3

# vmap over cosmo only; f_grid and astro are shared across the batch
batched_generate = jax.vmap(generate_signal, in_axes=(None, 0, None))


def make_batch_cosmo(batch_size: int) -> cosmology:
    """Return a batch of cosmologies with H0 linearly spaced over 60-74."""
    H0 = jnp.linspace(60.0, 74.0, batch_size)
    return cosmology(
        H0=H0,
        Omega_b=jnp.full(batch_size, planck.Omega_b),
        Omega_c=jnp.full(batch_size, planck.Omega_c),
        Y_He=jnp.full(batch_size, planck.Y_He),
        ns=jnp.full(batch_size, planck.ns),
        ln1010As=jnp.full(batch_size, planck.ln1010As),
    )


def benchmark_device(
    device: jax.Device,
    batch_sizes: list,
    n_repeats: int = N_REPEATS,
) -> list:
    """Return minimum wall time (s) for each batch size on device."""
    f = jax.device_put(f_grid, device)
    times = []
    for bs in batch_sizes:
        batch_cosmo = jax.device_put(make_batch_cosmo(bs), device)

        # Warm-up: triggers JIT compilation for this batch size
        sig, _, _ = batched_generate(f, batch_cosmo, astro)
        jax.block_until_ready(sig)

        t_runs = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            sig, _, _ = batched_generate(f, batch_cosmo, astro)
            jax.block_until_ready(sig)
            t_runs.append(time.perf_counter() - t0)

        best = min(t_runs)
        times.append(best)
        print(f"  batch={bs:5d}  {best:.3f} s  ({best/bs*1000:.1f} ms/signal)")

    return times


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
cpu = jax.devices("cpu")[0]
print(f"Benchmarking CPU ({cpu.device_kind})...")
cpu_times = benchmark_device(cpu, BATCH_SIZES)

results = {"CPU": (BATCH_SIZES, cpu_times)}

# ---------------------------------------------------------------------------
# GPU (optional)
# ---------------------------------------------------------------------------
gpu_device = None
if not skip_gpu:
    try:
        gpus = jax.devices("gpu")
        if gpus:
            gpu_device = gpus[0]
            print(f"\nBenchmarking GPU ({gpu_device.device_kind})...")
            gpu_times = benchmark_device(gpu_device, BATCH_SIZES)
            results["GPU"] = (BATCH_SIZES, gpu_times)
    except RuntimeError:
        print("No CUDA GPU found — skipping.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colours = {"CPU": "steelblue", "GPU": "darkorange"}

fig, (ax_total, ax_per) = plt.subplots(1, 2, figsize=(10, 4.5))

for label, (bs, times) in results.items():
    bs_arr = np.array(bs)
    t_arr = np.array(times)

    ax_total.plot(bs_arr, t_arr, "o-", color=colours[label], label=label)
    ax_per.plot(
        bs_arr, t_arr / bs_arr * 1000, "o-", color=colours[label], label=label
    )

# Ideal linear scaling reference from the single-signal CPU time
t1_cpu = cpu_times[0]
bs_ref = np.array(BATCH_SIZES)
ax_total.plot(
    bs_ref,
    t1_cpu * bs_ref,
    "--",
    color="grey",
    lw=1,
    label="Linear (CPU×1)",
)

for ax in (ax_total, ax_per):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.legend()
    ax.grid(which="both", ls=":", lw=0.5)

ax_total.set_ylabel("Wall time [s]")
ax_total.set_title("Total time vs batch size")

ax_per.set_ylabel("Time per signal [ms]")
ax_per.set_title("Throughput vs batch size")

plt.tight_layout()
out = Path(__file__).parent / "benchmark.png"
plt.savefig(out, dpi=150)
print(f"\nSaved {out}")
