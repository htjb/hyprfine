"""Tests for hyprfine.analytic.signal."""

import jax.numpy as jnp

from hyprfine.analytic.signal import T21
from hyprfine.parameters import cosmology


def test_t21_zero_when_no_contrast(cosmo: cosmology) -> None:
    """T21 should be zero when spin temperature equals CMB temperature."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    result = T21(z, jnp.array(50.0), T_cmb, T_cmb, jnp.array(1e-4), cosmo)
    assert jnp.isclose(result, 0.0, atol=1e-6)


def test_t21_absorption_when_ts_below_tcmb(cosmo: cosmology) -> None:
    """T21 should be negative (absorption) when Ts < Tcmb — the dark ages."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    T_s = jnp.array(50.0)  # gas cooler than CMB
    result = T21(z, T_s, T_cmb, T_s, jnp.array(1e-4), cosmo)
    assert result < 0


def test_t21_emission_when_ts_above_tcmb(cosmo: cosmology) -> None:
    """T21 should be positive (emission) when Ts > Tcmb."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    T_s = jnp.array(500.0)  # gas hotter than CMB
    result = T21(z, T_s, T_cmb, T_s, jnp.array(1e-4), cosmo)
    assert result > 0


def test_t21_magnitude_reasonable(cosmo: cosmology) -> None:
    """Dark ages signal should be of order tens of mK."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    T_s = jnp.array(30.0)
    result = T21(z, T_s, T_cmb, T_s, jnp.array(1e-4), cosmo)
    assert -500.0 < float(result) < 0.0
