"""Tests for hyprfine.analytic.xrays."""

import jax.numpy as jnp

from hyprfine.analytic.xrays import (
    calculate_epsilon_x_intrinsic,
    sigma_X,
    tau_X,
)
from hyprfine.parameters import astrophysics, const, cosmology


def test_sigma_x_zero_below_threshold() -> None:
    """Photoionisation cross section should be zero below the HI threshold."""
    nu_below = jnp.array([1e14, 1e15])
    assert jnp.all(sigma_X(nu_below) == 0.0)


def test_sigma_x_positive_above_threshold() -> None:
    """Photoionisation cross section should be positive above the HI threshold."""
    nu_above = jnp.array([1e16, 1e17, 1e18])
    assert jnp.all(sigma_X(nu_above) > 0)


def test_sigma_x_decreases_with_frequency() -> None:
    """Cross section should decrease as nu^-3 with increasing frequency."""
    nu = jnp.array([1e16, 1e17])
    sig = sigma_X(nu)
    assert sig[0] > sig[1]
    ratio = sig[0] / sig[1]
    expected = (nu[1] / nu[0]) ** 3
    assert jnp.isclose(ratio, expected, rtol=1e-4)


def test_epsilon_x_intrinsic_zero_outside_band(astro: astrophysics) -> None:
    """X-ray emissivity should be zero outside the 0.5–2 keV band."""
    from hyprfine.parameters import conv
    nu_below = jnp.array([0.1 * conv.keV_to_Hz])
    nu_above = jnp.array([5.0 * conv.keV_to_Hz])
    assert jnp.all(calculate_epsilon_x_intrinsic(nu_below, astro) == 0.0)
    assert jnp.all(calculate_epsilon_x_intrinsic(nu_above, astro) == 0.0)


def test_epsilon_x_intrinsic_positive_in_band(astro: astrophysics) -> None:
    """X-ray emissivity should be positive within the 0.5–2 keV band."""
    from hyprfine.parameters import conv
    nu_in_band = jnp.linspace(0.6 * conv.keV_to_Hz, 1.9 * conv.keV_to_Hz, 20)
    result = calculate_epsilon_x_intrinsic(nu_in_band, astro)
    assert jnp.any(result > 0)


def test_epsilon_x_intrinsic_scales_with_l40(cosmo: cosmology) -> None:
    """X-ray emissivity should scale with L40."""
    from hyprfine.parameters import conv
    nu = jnp.linspace(0.6 * conv.keV_to_Hz, 1.9 * conv.keV_to_Hz, 20)
    astro1 = astrophysics(L40=1.0)
    astro2 = astrophysics(L40=10.0)
    eps1 = calculate_epsilon_x_intrinsic(nu, astro1)
    eps2 = calculate_epsilon_x_intrinsic(nu, astro2)
    assert jnp.all(eps2 > eps1)


def test_tau_x_positive(cosmo: cosmology) -> None:
    """X-ray optical depth should be non-negative."""
    nu_obs = jnp.array(1e17)
    result = tau_X(nu_obs, jnp.array(10.0), jnp.array(20.0), cosmo)
    assert result >= 0


def test_tau_x_increases_with_path_length(cosmo: cosmology) -> None:
    """Optical depth should increase with the source-observer separation."""
    nu_obs = jnp.array(1e17)
    tau_short = tau_X(nu_obs, jnp.array(10.0), jnp.array(15.0), cosmo)
    tau_long = tau_X(nu_obs, jnp.array(10.0), jnp.array(25.0), cosmo)
    assert tau_long > tau_short
