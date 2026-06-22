"""Tests for hyprfine.analytic.wouthuysen_field."""

import jax.numpy as jnp

from hyprfine.analytic.wouthuysen_field import (
    calculate_epsilon_alpha_intrinsic,
    calculate_epsilon_alpha_tot,
    get_f_rec,
)
from hyprfine.parameters import astrophysics, const


def test_f_rec_direct_lyman_alpha() -> None:
    """Recycling fraction for n=2 (direct Ly-alpha) should be 1."""
    assert jnp.isclose(get_f_rec(2), 1.0)


def test_f_rec_lyman_beta_zero() -> None:
    """Recycling fraction for n=3 (Ly-beta) should be 0."""
    assert jnp.isclose(get_f_rec(3), 0.0)


def test_f_rec_higher_order_between_zero_and_one() -> None:
    """Recycling fractions for n>=4 should be between 0 and 1."""
    for n in [4, 5, 10, 23]:
        f = get_f_rec(n)
        assert 0.0 < float(f) < 1.0


def test_f_rec_asymptotes(astro: astrophysics) -> None:
    """Recycling fractions should converge toward ~0.36 at high n."""
    f_high = get_f_rec(23)
    assert 0.3 < float(f_high) < 0.4


def test_epsilon_alpha_intrinsic_zero_below_lya(astro: astrophysics) -> None:
    """Emissivity should be zero below Lyman-alpha frequency."""
    nu_below = jnp.array([1e15, 2e15])
    result = calculate_epsilon_alpha_intrinsic(nu_below, astro)
    assert jnp.all(result == 0.0)


def test_epsilon_alpha_intrinsic_positive_in_band(astro: astrophysics) -> None:
    """Emissivity should be positive between Ly-alpha and Lyman limit."""
    nu_in_band = jnp.linspace(
        const.lyman_alpha_freq * 1.01, const.lyman_limit * 0.99, 50
    )
    result = calculate_epsilon_alpha_intrinsic(nu_in_band, astro)
    assert jnp.any(result > 0)


def test_epsilon_alpha_intrinsic_break_at_lybeta(astro: astrophysics) -> None:
    """Emissivity should drop steeply above Ly-beta (alpha_high << 0)."""
    nu_low = jnp.array([const.lyman_alpha_freq * 1.05])
    nu_high = jnp.array([const.lyman_beta_freq * 1.05])
    eps_low = calculate_epsilon_alpha_intrinsic(nu_low, astro)
    eps_high = calculate_epsilon_alpha_intrinsic(nu_high, astro)
    assert eps_high < eps_low


def test_epsilon_alpha_tot_zero_for_source_below_observer(
    astro: astrophysics,
) -> None:
    """Total emissivity should be zero when source redshift < observer redshift."""
    result = calculate_epsilon_alpha_tot(
        z_source=jnp.array(5.0), z_21=jnp.array(10.0), astro=astro
    )
    assert jnp.all(result == 0.0)


def test_epsilon_alpha_tot_positive_for_source_above_observer(
    astro: astrophysics,
) -> None:
    """Total emissivity should be positive when source is at higher redshift."""
    result = calculate_epsilon_alpha_tot(
        z_source=jnp.array(15.0), z_21=jnp.array(10.0), astro=astro
    )
    assert jnp.any(result > 0)
