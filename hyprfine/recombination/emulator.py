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
    z_grid_orig = jnp.asarray(z_grid, dtype=jnp.float32)
    results = []
    for label in ("xe", "tk"):
        loaded = _load_emulator(label)
        pipeline = loaded["train_pipeline"]

        # Add batch dim so grid-redistribution pipeline steps (which expect
        # shape (batch, len_x)) work correctly during inference.
        x = z_grid_orig[None, :]  # (1, len_z)
        params = jnp.array(
            [H0, omb, omc, yhe], dtype=jnp.float32
        )[None, :]   # (1, 4)
        y_dummy = jnp.ones_like(x)

        for pipe in pipeline:
            y_dummy, x, params = pipe.forward(y_dummy, x, params)

        # x is now (1, n_training_grid) — flatten for MLP input.
        x_flat = x[0]       # (n_training,)
        params_flat = params[0]  # (4,)

        # Build tiled input: each row is [z_norm_i, param0_norm, ...]
        tiled = jnp.column_stack(
            [x_flat, jnp.tile(params_flat, (len(x_flat), 1))]
        )
        preds = mlp(
            loaded["params"], tiled, act=loaded["hyperparams"]["act"]
        )

        # Backward pass to recover physical units on the training grid.
        preds = preds.reshape(1, -1)
        x_back = x
        for pipe in reversed(pipeline):
            preds, x_back, params = pipe.backward(preds, x_back, params)

        # x_back is now (1, n_training_grid) in original z-space.
        # Interpolate to the user's requested z_grid.
        z_train = x_back[0]
        result = jnp.interp(z_grid_orig, z_train, preds.reshape(-1))
        results.append(result)

    return results[0], results[1]
