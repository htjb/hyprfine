"""Benchmark hyprfine signal generation on CPU (and CUDA GPU if available)."""

import platform
import subprocess
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


def get_cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            for key in ("machdep.cpu.brand_string", "hw.chip_name"):
                try:
                    name = subprocess.check_output(
                        ["sysctl", "-n", key], stderr=subprocess.DEVNULL,
                    ).decode().strip()
                    if name:
                        return name
                except subprocess.CalledProcessError:
                    continue
        elif system == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "CPU"


batch_sizes = [1, 2, 4, 8, 16, 32, 64]
batched_generate = jax.vmap(generate_signal, in_axes=(None, 0, 0))
cpu_name = get_cpu_name()


def make_batched_params(n: int) -> tuple:
    bc = cosmology(
        H0=jnp.full(n, H0),
        Omega_b=jnp.full(n, Omega_b),
        Omega_c=jnp.full(n, Omega_c),
        Y_He=jnp.full(n, Y_He),
        ns=jnp.full(n, ns),
        ln1010As=jnp.full(n, ln1010As),
    )
    ba = astrophysics(
        epsilon=jnp.full(n, epsilon),
        alpha_star=jnp.full(n, alpha_star),
        beta_star=jnp.full(n, beta_star),
        M_pivot=jnp.full(n, M_pivot),
        L40=jnp.full(n, L40),
        alpha_x=jnp.full(n, alpha_x),
        nu_0=jnp.full(n, E0_keV),
        alpha_low=jnp.full(n, alpha_low),
        alpha_high=jnp.full(n, alpha_high),
        N_alpha=jnp.full(n, N_alpha),
        f_esc=jnp.full(n, f_esc),
        N_ion=jnp.full(n, N_ion),
    )
    return bc, ba


def time_device(device: jax.Device) -> tuple[float, float]:
    f = jax.device_put(f_grid, device)

    t0 = time.perf_counter()
    sig, xe, T_gas = generate_signal(f, planck, astro)
    jax.block_until_ready(sig)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    sig, xe, T_gas = generate_signal(f, planck, astro)
    jax.block_until_ready(sig)
    warm = time.perf_counter() - t0

    return cold, warm, sig


def bench_batched(device: jax.Device) -> list[float]:
    """Return warm time-per-signal for each batch size."""
    f = jax.device_put(f_grid, device)
    times = []
    for bs in batch_sizes:
        bc, ba = make_batched_params(bs)
        bc = jax.device_put(bc, device)
        ba = jax.device_put(ba, device)
        # cold — includes compilation for this batch size
        sigs, _, _ = batched_generate(f, bc, ba)
        jax.block_until_ready(sigs)
        # warm
        t0 = time.perf_counter()
        sigs, _, _ = batched_generate(f, bc, ba)
        jax.block_until_ready(sigs)
        times.append((time.perf_counter() - t0) / bs)
        print(f"    batch={bs:3d}  {times[-1]:.3f} s/signal")
    return times


# ---------------------------------------------------------------------------
# Always benchmark CPU
# ---------------------------------------------------------------------------
cpu = jax.devices("cpu")[0]
print("Benchmarking CPU...")
cpu_cold, cpu_warm, signal = time_device(cpu)
print(f"  Cold: {cpu_cold:.3f} s   Warm: {cpu_warm:.3f} s")
print("  Batch benchmark (CPU):")
cpu_batch_times = bench_batched(cpu)

# ---------------------------------------------------------------------------
# Optionally benchmark CUDA GPU
# ---------------------------------------------------------------------------
gpu_batch_times = None
try:
    if not skip_gpu:
        gpus = jax.devices("gpu")
        if gpus:
            gpu = gpus[0]
            print(f"Benchmarking GPU ({gpu.device_kind})...")
            gpu_cold, gpu_warm, _ = time_device(gpu)
            print(f"  Cold: {gpu_cold:.3f} s   Warm: {gpu_warm:.3f} s")
            print(f"  Batch benchmark (GPU):")
            gpu_batch_times = bench_batched(gpu)
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

# ---------------------------------------------------------------------------
# Plot: signal + time-per-signal vs batch size
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax_sig = fig.add_subplot(gs[0, :2])
ax_batch = fig.add_subplot(gs[0, 2])

# Signal
z_grid = 1420.4 / np.array(f_grid) - 1
ax_sig.plot(
    z_grid, np.array(signal), color="steelblue", lw=1.5, label="hyprfine",
)
ax_sig.plot(
    T21c.zintegral, T21c.T21avg,
    color="seagreen", lw=1.5, ls="--", label="zeus21",
)
ax_sig.set_xlabel("Redshift $z$")
ax_sig.set_ylabel(r"$T_{21}$ [mK]")
ax_sig.set_title("21-cm signal")
ax_sig.set_xscale("log")
ax_sig.legend(fontsize=8)

# Time-per-signal vs batch size
ax_batch.plot(
    batch_sizes, cpu_batch_times,
    color="steelblue", marker="o", lw=1.5, label=f"hyprfine ({cpu_name})",
)
if gpu_batch_times is not None:
    ax_batch.plot(
        batch_sizes, gpu_batch_times,
        color="darkorange", marker="s", lw=1.5,
        label=f"hyprfine ({gpu.device_kind})",
    )
ax_batch.axhline(
    z21_time, color="seagreen", ls="--", lw=1.5, label="zeus21 (CPU)",
)
ax_batch.set_xlabel("Batch size")
ax_batch.set_ylabel("Time per signal [s]")
ax_batch.set_title("Throughput scaling")
ax_batch.set_xscale("log", base=2)
ax_batch.set_yscale("log")
ax_batch.set_xticks(batch_sizes)
ax_batch.set_xticklabels([str(b) for b in batch_sizes])
ax_batch.legend(fontsize=8)

plt.savefig("bin/benchmark.png", dpi=300)
print("Saved benchmark.png")
