"""Emulator-based recombination, replacing the HYREC-2 C-code wrapper."""

from pathlib import Path

import jax.numpy as jnp
from astroemu.network import mlp
from astroemu.serialisation import load

_DATA_DIR = Path(__file__).parent.parent / "data"

# Module-level cache: populated on first call, reused for every subsequent
# cosmology evaluation within the same process.
_cache: dict = {}


def _load_emulator(label: str) -> dict:
    """Load and cache an emulator by label.

    Args:
        label: One of 'xe' or 'tk'.

    Returns:
        Loaded emulator dict with keys 'params', 'hyperparams',
        'train_pipeline', etc.
    """
    if label not in _cache:
        path = _DATA_DIR / f"hyrec_{label}.astroemu"
        _cache[label] = load(str(path))
    return _cache[label]


def call_hyrec_emulator(
    z_grid: jnp.ndarray,
    H0: float,
    omb: float,
    omc: float,
    yhe: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Emulate xe(z) and Tk(z) using trained neural network emulators.

    Replaces the set_up_hyrec / call_hyrec pair. Emulators are loaded
    once and cached for the lifetime of the process.

    Args:
        z_grid: Redshift values to evaluate on.
        H0: Hubble constant in km/s/Mpc.
        omb: Baryon density parameter Omega_b.
        omc: Cold dark matter density parameter Omega_c.
        yhe: Helium mass fraction Y_He.

    Returns:
        xe: Free electron fraction at each redshift in z_grid.
        Tk: Gas temperature in Kelvin at each redshift in z_grid.
    """
    results = []
    for label in ("xe", "tk"):
        loaded = _load_emulator(label)
        pipeline = loaded["train_pipeline"]

        # Walk the forward pipeline to get normalised x and params.
        # y_dummy is required by the pipeline interface but is not used.
        x = jnp.asarray(z_grid, dtype=jnp.float32)
        params = jnp.array([H0, omb, omc, yhe], dtype=jnp.float32)
        y_dummy = jnp.ones_like(x)
        for pipe in pipeline:
            y_dummy, x, params = pipe.forward(y_dummy, x, params)

        # Build tiled input matching the training format:
        # each row is [normalised_z_i, normalised_H0, normalised_omb, ...]
        tiled = jnp.column_stack([x, jnp.tile(params, (len(x), 1))])

        preds = mlp(
            loaded["params"], tiled, act=loaded["hyperparams"]["act"]
        )

        # Reshape to (1, n_z) before the backward pass so per-frequency
        # statistics (shape len_z,) broadcast correctly.
        preds = preds.reshape(1, -1)
        for pipe in reversed(pipeline):
            preds, _, _ = pipe.backward(preds, x, params)

        results.append(preds.reshape(-1))

    return results[0], results[1]
