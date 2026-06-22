"""Shared fixtures for hyprfine tests."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import pytest  # noqa: E402

from hyprfine.parameters import astrophysics, cosmology

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def cosmo() -> cosmology:
    """Planck 2018 cosmology."""
    return cosmology(
        H0=67.32,
        Omega_b=0.0494,
        Omega_c=0.2664,
        Y_He=0.2454,
    )


@pytest.fixture
def astro() -> astrophysics:
    """Default astrophysics parameters."""
    return astrophysics()
