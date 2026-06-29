"""Run hyprfine/zeus21 timing benchmarks and save results to a JSON file.

Usage:
    python bin/benchmark_time.py [--output results.json] [--no-gpu]

Output filename defaults to bin/benchmark_<cpu_name>.json.
"""

import argparse
import json
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
import numpy as np
import zeus21

from hyprfine.analytic.main import generate_signal
from hyprfine.parameters import astrophysics, cosmology

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=None)
parser.add_argument("--no-gpu", action="store_true")
args = parser.parse_args()

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

f_grid = jnp.linspace(5.0, 250.0, 500)  # MHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_cpu_name() -> str:
    """Get a human-readable CPU name."""
    system = platform.system()
    try:
        if system == "Darwin":
            for key in ("machdep.cpu.brand_string", "hw.chip_name"):
                try:
                    name = (
                        subprocess.check_output(
                            ["sysctl", "-n", key],
                            stderr=subprocess.DEVNULL,
                        )
                        .decode()
                        .strip()
                    )
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


def make_batched_params(n: int) -> tuple[cosmology, astrophysics]:
    """Create batched cosmology and astrophysics parameters.

    Args:
        n: Number of samples to create.

    Returns:
        Tuple of (cosmology, astrophysics) with n samples each.
    """
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


def time_device(device: jax.Device) -> tuple[float, float, jnp.ndarray]: # type: ignore
    """Time the generate_signal function on a given device.

    Args:
        device: JAX device to run the benchmark on.

    Returns:
        Tuple of (cold_time, warm_time, signal) where cold_time is the time
            for the first run, warm_time is the time for the second run,
            and signal is the output of generate_signal.
    """
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


def bench_batched(device: jax.Device) -> list: # type: ignore
    """Benchmark batched generate_signal on a given device.

    Args:
        device: JAX device to run the benchmark on.
    
    Returns:
        List of times per signal for each batch size.
    """
    f = jax.device_put(f_grid, device)
    times = []
    for bs in batch_sizes:
        bc, ba = make_batched_params(bs)
        bc = jax.device_put(bc, device)
        ba = jax.device_put(ba, device)
        sigs, _, _ = batched_generate(f, bc, ba)
        jax.block_until_ready(sigs)
        t0 = time.perf_counter()
        sigs, _, _ = batched_generate(f, bc, ba)
        jax.block_until_ready(sigs)
        times.append((time.perf_counter() - t0) / bs)
        print(f"    batch={bs:3d}  {times[-1]:.3f} s/signal")
    return times


# ---------------------------------------------------------------------------
# CPU benchmark
# ---------------------------------------------------------------------------
cpu = jax.devices("cpu")[0]
print(f"Benchmarking CPU ({cpu_name})...")
cpu_cold, cpu_warm, signal = time_device(cpu)
print(f"  Cold: {cpu_cold:.3f} s   Warm: {cpu_warm:.3f} s")
print("  Batch benchmark:")
cpu_batch_times = bench_batched(cpu)

# ---------------------------------------------------------------------------
# GPU benchmark
# ---------------------------------------------------------------------------
gpu_name = None
gpu_batch_times = None
try:
    if not args.no_gpu:
        gpus = jax.devices("gpu")
        if gpus:
            gpu = gpus[0]
            gpu_name = gpu.device_kind
            print(f"Benchmarking GPU ({gpu_name})...")
            gpu_cold, gpu_warm, _ = time_device(gpu)
            print(f"  Cold: {gpu_cold:.3f} s   Warm: {gpu_warm:.3f} s")
            print("  Batch benchmark:")
            gpu_batch_times = bench_batched(gpu)
except RuntimeError:
    print("No CUDA GPU found — skipping GPU benchmark.")

# ---------------------------------------------------------------------------
# zeus21 benchmark
# ---------------------------------------------------------------------------
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

print("Benchmarking zeus21 (CPU)...")
t0 = time.perf_counter()
T21c = zeus21.get_T21_coefficients(
    user_params,
    CosmoParams,
    ClassyCosmo,
    astro_params,
    HMFintclass,
    zmin=10.0,
)
z21_time = time.perf_counter() - t0
print(f"  {z21_time:.3f} s")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output = args.output
if output is None:
    safe = cpu_name.replace(" ", "_").replace("/", "-")
    output = f"bin/benchmark_{safe}.json"

data = {
    "cpu_name": cpu_name,
    "gpu_name": gpu_name,
    "batch_sizes": batch_sizes,
    "cpu_batch_times": cpu_batch_times,
    "gpu_batch_times": gpu_batch_times,
    "z21_time": z21_time,
    "signal": np.array(signal).tolist(),
    "f_grid": np.array(f_grid).tolist(),
    "z21_signal": np.array(T21c.T21avg).tolist(),
    "z21_zintegral": np.array(T21c.zintegral).tolist(),
}

with open(output, "w") as fh:
    json.dump(data, fh, indent=2)
print(f"Saved {output}")
