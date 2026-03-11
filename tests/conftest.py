"""Shared fixtures for hyprfine tests."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest

from hyprfine.parameters import cosmology


@pytest.fixture
def cosmo() -> cosmology:
    """Planck 2018 cosmology."""
    return cosmology(
        H0=67.32,
        Omega_m=0.3158,
        Omega_b=0.0494,
        Omega_c=0.2664,
        Y_He=0.2454,
    )
