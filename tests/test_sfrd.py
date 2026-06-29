"""Tests for hyprfine.analytic.sfrd."""

import jax.numpy as jnp

from hyprfine.analytic.sfrd import dn_dmh, fstar, mean_sfrd
from hyprfine.parameters import astrophysics, cosmology


def test_fstar_positive(cosmo: cosmology, astro: astrophysics) -> None:
    """Star formation efficiency should be positive."""
    Mh = jnp.array(1e10)
    assert fstar(astro, Mh, jnp.array(10.0)) > 0


def test_fstar_suppressed_below_mturn(
    cosmo: cosmology, astro: astrophysics
) -> None:
    """Fstar should be suppressed for halos well below the turnover mass."""
    z = jnp.array(20.0)
    f_large = fstar(astro, jnp.array(1e12), z)
    f_small = fstar(astro, jnp.array(1e5), z)
    assert f_large > f_small


def test_fstar_scales_with_epsilon(cosmo: cosmology) -> None:
    """Fstar should scale proportionally with epsilon."""
    Mh = jnp.array(1e10)
    z = jnp.array(10.0)
    astro1 = astrophysics(epsilon=0.1)
    astro2 = astrophysics(epsilon=0.2)
    assert fstar(astro2, Mh, z) > fstar(astro1, Mh, z)


def test_dn_dmh_positive(cosmo: cosmology) -> None:
    """Halo mass function should be positive."""
    Mh = jnp.logspace(8, 13, 50)
    result = dn_dmh(Mh, cosmo, jnp.array(10.0))
    assert jnp.all(result > 0)


def test_dn_dmh_decreases_at_high_mass(cosmo: cosmology) -> None:
    """Halo mass function should decrease toward high masses."""
    Mh = jnp.logspace(8, 15, 100)
    result = dn_dmh(Mh, cosmo, jnp.array(10.0))
    assert result[-1] < result[0]


def test_mean_sfrd_positive(cosmo: cosmology, astro: astrophysics) -> None:
    """Mean SFRD should be positive."""
    Mh = jnp.logspace(8, 13, 50)
    result = mean_sfrd(jnp.array(10.0), Mh, astro, cosmo)
    assert result > 0


def test_mean_sfrd_increases_with_epsilon(cosmo: cosmology) -> None:
    """Higher star formation efficiency should give higher SFRD."""
    Mh = jnp.logspace(8, 13, 50)
    z = jnp.array(10.0)
    sfrd_low = mean_sfrd(z, Mh, astrophysics(epsilon=0.05), cosmo)
    sfrd_high = mean_sfrd(z, Mh, astrophysics(epsilon=0.2), cosmo)
    assert sfrd_high > sfrd_low
