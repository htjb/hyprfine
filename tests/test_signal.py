"""Tests for hyprfine.analytic.signal and hyprfine.analytic.main."""

import jax.numpy as jnp

from hyprfine.analytic.main import generate_signal
from hyprfine.analytic.signal import T21
from hyprfine.parameters import astrophysics, cosmology


def test_t21_zero_when_no_contrast(cosmo: cosmology) -> None:
    """T21 should be zero when spin temperature equals CMB temperature."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    result = T21(z, T_cmb, T_cmb, jnp.array(1e-4), cosmo)
    assert jnp.isclose(result, 0.0, atol=1e-6)


def test_t21_absorption_when_ts_below_tcmb(cosmo: cosmology) -> None:
    """T21 should be negative (absorption) when Ts < Tcmb."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    T_s = jnp.array(50.0)
    result = T21(z, T_cmb, T_s, jnp.array(1e-4), cosmo)
    assert result < 0


def test_t21_emission_when_ts_above_tcmb(cosmo: cosmology) -> None:
    """T21 should be positive (emission) when Ts > Tcmb."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    T_s = jnp.array(500.0)
    result = T21(z, T_cmb, T_s, jnp.array(1e-4), cosmo)
    assert result > 0


def test_t21_magnitude_reasonable(cosmo: cosmology) -> None:
    """Dark ages signal should be of order tens to hundreds of mK."""
    z = jnp.array(50.0)
    T_cmb = jnp.array(137.5)
    T_s = jnp.array(30.0)
    result = T21(z, T_cmb, T_s, jnp.array(1e-4), cosmo)
    assert -500.0 < float(result) < 0.0


def test_generate_signal_dark_ages_shape(cosmo: cosmology) -> None:
    """generate_signal without astro should return arrays matching f_grid."""
    f_grid = jnp.linspace(10, 50, 100)
    T21_out, xe, Tk = generate_signal(f_grid, cosmo)
    assert T21_out.shape == f_grid.shape
    assert xe.shape == f_grid.shape
    assert Tk.shape == f_grid.shape


def test_generate_signal_dark_ages_absorption(cosmo: cosmology) -> None:
    """Dark ages signal should be in absorption (negative T21)."""
    f_grid = jnp.linspace(10, 50, 50)
    T21_out, _, _ = generate_signal(f_grid, cosmo)
    assert jnp.all(T21_out < 0)


def test_generate_signal_dark_ages_finite(cosmo: cosmology) -> None:
    """Dark ages signal should be finite everywhere."""
    f_grid = jnp.linspace(10, 50, 50)
    T21_out, xe, Tk = generate_signal(f_grid, cosmo)
    assert jnp.all(jnp.isfinite(T21_out))
    assert jnp.all(jnp.isfinite(xe))
    assert jnp.all(jnp.isfinite(Tk))


def test_generate_signal_cosmic_dawn_shape(
    cosmo: cosmology, astro: astrophysics
) -> None:
    """generate_signal with astro should return arrays matching f_grid."""
    f_grid = jnp.linspace(50, 200, 50)
    T21_out, xe, Tk = generate_signal(f_grid, cosmo, astro)
    assert T21_out.shape == f_grid.shape
    assert xe.shape == f_grid.shape
    assert Tk.shape == f_grid.shape


def test_generate_signal_cosmic_dawn_finite(
    cosmo: cosmology, astro: astrophysics
) -> None:
    """Cosmic Dawn signal should be finite everywhere."""
    f_grid = jnp.linspace(50, 200, 50)
    T21_out, xe, Tk = generate_signal(f_grid, cosmo, astro)
    assert jnp.all(jnp.isfinite(T21_out))
    assert jnp.all(jnp.isfinite(xe))
    assert jnp.all(jnp.isfinite(Tk))


def test_generate_signal_cosmic_dawn_xe_bounded(
    cosmo: cosmology, astro: astrophysics
) -> None:
    """Free electron fraction should stay in (0, 1] during Cosmic Dawn."""
    f_grid = jnp.linspace(50, 200, 50)
    _, xe, _ = generate_signal(f_grid, cosmo, astro)
    assert jnp.all(xe > 0)
    assert jnp.all(xe <= 1.0)


def test_generate_signal_cosmic_dawn_tk_positive(
    cosmo: cosmology, astro: astrophysics
) -> None:
    """Gas temperature should be positive during Cosmic Dawn."""
    f_grid = jnp.linspace(50, 200, 50)
    _, _, Tk = generate_signal(f_grid, cosmo, astro)
    assert jnp.all(Tk > 0)
