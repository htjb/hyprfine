"""Tests for hyprfine.analytic.coupling_coeffs."""

import jax.numpy as jnp

from hyprfine.analytic.coupling_coeffs import kappa, xc
from hyprfine.parameters import cosmology


def test_kappa_positive() -> None:
    """Kappa should be positive."""
    assert kappa(50.0, jnp.array(100.0), jnp.array(1e-4)) > 0


def test_kappa_neutral_gas() -> None:
    """In neutral gas (xe=0) kappa should be dominated by H-H collisions."""
    k_neutral = kappa(50.0, jnp.array(100.0), jnp.array(0.0))
    k_ionised = kappa(50.0, jnp.array(100.0), jnp.array(1.0))
    assert k_neutral != k_ionised


def test_kappa_increases_with_temperature() -> None:
    """Kappa should increase with gas temperature."""
    k_cold = kappa(50.0, jnp.array(10.0), jnp.array(1e-4))
    k_warm = kappa(50.0, jnp.array(1000.0), jnp.array(1e-4))
    assert k_warm > k_cold


def test_xc_positive_and_increases_with_density(cosmo: cosmology) -> None:
    """Xc should be positive and larger at higher redshift (higher density)."""
    xc_low_z = xc(jnp.array(50.0), jnp.array(1e-4), jnp.array(100.0), cosmo)
    xc_high_z = xc(jnp.array(200.0), jnp.array(1e-4), jnp.array(200.0), cosmo)
    assert xc_high_z > 0
    assert xc_high_z > xc_low_z
