"""Benchmark hyprfine signal generation on CPU (and CUDA GPU if available).

Dark Ages (no ODE): vmapped over a batch of cosmologies — shows GPU scaling.
Cosmic Dawn (ODE + J_X): single-signal timing only — the nested vmaps inside
the ODE solver make full batching too memory-intensive with the current code.
"""

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

BATCH_SIZES = [1, 10, 50, 100, 500]
N_REPEATS = 3

# vmap dark-ages generate_signal (no astro) over a batch of cosmologies
batched_dark_ages = jax.vmap(generate_signal, in_axes=(None, 0, None))


def make_batch_cosmo(batch_size: int) -> cosmology:
    """Return a batch of cosmologies with H0 linearly spaced over 60–74."""
    H0 = jnp.linspace(60.0, 74.0, batch_size)
    return cosmology(
        H0=H0,
        Omega_b=jnp.full(batch_size, planck.Omega_b),
        Omega_c=jnp.full(batch_size, planck.Omega_c),
        Y_He=jnp.full(batch_size, planck.Y_He),
        ns=jnp.full(batch_size, planck.ns),
        ln1010As=jnp.full(batch_size, planck.ln1010As),
    )


def time_fn(fn, *args, n_repeats=N_REPEATS):
    """Return minimum wall time over n_repeats warm calls."""
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times)


def benchmark_da_batch(device, batch_sizes):
    """Batch dark-ages benchmark (vmapped over cosmologies)."""
    f = jax.device_put(f_grid, device)
    times = []
    for bs in batch_sizes:
        bc = jax.device_put(make_batch_cosmo(bs), device)
        t = time_fn(batched_dark_ages, f, bc, None)
        times.append(t)
        print(f"  batch={bs:5d}  {t:.3f} s  ({t/bs*1000:.1f} ms/signal)")
    return times


def benchmark_single(device, mode="dark_ages"):
    """Single-signal benchmark — cold and warm."""
    f = jax.device_put(f_grid, device)
    bc = jax.device_put(make_batch_cosmo(1), device)

    if mode == "dark_ages":
        fn = lambda: batched_dark_ages(f, bc, None)  # noqa: E731
    else:
        fn = lambda: generate_signal(f, planck, astro)  # noqa: E731

    # cold (includes JIT)
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    cold = time.perf_counter() - t0

    warm = time_fn(fn)
    return cold, warm


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
cpu = jax.devices("cpu")[0]
print(f"\n── CPU ({cpu.device_kind}) ──")

print("Single-signal (dark ages):")
cpu_da_cold, cpu_da_warm = benchmark_single(cpu, "dark_ages")
print(f"  cold={cpu_da_cold:.3f} s   warm={cpu_da_warm:.3f} s")

print("Single-signal (cosmic dawn):")
cpu_cd_cold, cpu_cd_warm = benchmark_single(cpu, "cosmic_dawn")
print(f"  cold={cpu_cd_cold:.3f} s   warm={cpu_cd_warm:.3f} s")

print("Batched dark ages:")
cpu_batch_times = benchmark_da_batch(cpu, BATCH_SIZES)

results_single = {"CPU": (cpu_da_warm, cpu_cd_warm)}
results_batch = {"CPU": (BATCH_SIZES, cpu_batch_times)}

# ---------------------------------------------------------------------------
# GPU (optional)
# ---------------------------------------------------------------------------
gpu_device = None
if not skip_gpu:
    try:
        gpus = jax.devices("gpu")
        if gpus:
            gpu_device = gpus[0]
            print(f"\n── GPU ({gpu_device.device_kind}) ──")

            print("Single-signal (dark ages):")
            gpu_da_cold, gpu_da_warm = benchmark_single(gpu_device, "dark_ages")
            print(f"  cold={gpu_da_cold:.3f} s   warm={gpu_da_warm:.3f} s")

            print("Single-signal (cosmic dawn):")
            gpu_cd_cold, gpu_cd_warm = benchmark_single(
                gpu_device, "cosmic_dawn"
            )
            print(f"  cold={gpu_cd_cold:.3f} s   warm={gpu_cd_warm:.3f} s")

            print("Batched dark ages:")
            gpu_batch_times = benchmark_da_batch(gpu_device, BATCH_SIZES)

            results_single["GPU"] = (gpu_da_warm, gpu_cd_warm)
            results_batch["GPU"] = (BATCH_SIZES, gpu_batch_times)
    except RuntimeError:
        print("No CUDA GPU found — skipping.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colours = {"CPU": "steelblue", "GPU": "darkorange"}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
ax_bar, ax_total, ax_per = axes

# --- Left: single-signal bar chart (dark ages vs cosmic dawn) ---
labels, da_vals, cd_vals = [], [], []
for dev, (da, cd) in results_single.items():
    labels.append(dev)
    da_vals.append(da)
    cd_vals.append(cd)

x = np.arange(len(labels))
w = 0.35
bars1 = ax_bar.bar(x - w / 2, da_vals, w, label="Dark Ages", color="steelblue")
bars2 = ax_bar.bar(x + w / 2, cd_vals, w, label="Cosmic Dawn", color="darkorange")
for bar, v in [(b, val) for bars, vals in [(bars1, da_vals), (bars2, cd_vals)]
               for b, val in zip(bars, vals)]:
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.02,
        f"{v:.2f}s",
        ha="center", va="bottom", fontsize=8,
    )
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels)
ax_bar.set_ylabel("Warm time [s]")
ax_bar.set_title("Single-signal timing")
ax_bar.legend(fontsize=8)
ax_bar.grid(axis="y", ls=":", lw=0.5)

# --- Middle: total batch time ---
for dev, (bs, times) in results_batch.items():
    ax_total.plot(bs, times, "o-", color=colours[dev], label=dev)

t1_cpu = cpu_batch_times[0]
bs_ref = np.array(BATCH_SIZES)
ax_total.plot(bs_ref, t1_cpu * bs_ref, "--", color="grey", lw=1,
              label="Linear (CPU×1)")
ax_total.set_xscale("log")
ax_total.set_yscale("log")
ax_total.set_xlabel("Batch size")
ax_total.set_ylabel("Wall time [s]")
ax_total.set_title("Batched dark ages — total time")
ax_total.legend(fontsize=8)
ax_total.grid(which="both", ls=":", lw=0.5)

# --- Right: time per signal ---
for dev, (bs, times) in results_batch.items():
    bs_arr = np.array(bs)
    t_arr = np.array(times)
    ax_per.plot(bs_arr, t_arr / bs_arr * 1000, "o-", color=colours[dev],
                label=dev)
ax_per.set_xscale("log")
ax_per.set_yscale("log")
ax_per.set_xlabel("Batch size")
ax_per.set_ylabel("Time per signal [ms]")
ax_per.set_title("Batched dark ages — throughput")
ax_per.legend(fontsize=8)
ax_per.grid(which="both", ls=":", lw=0.5)

plt.tight_layout()
out = Path(__file__).parent / "benchmark.png"
plt.savefig(out, dpi=150)
print(f"\nSaved {out}")
