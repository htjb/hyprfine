"""Tests for hyprfine.utils.cosmology."""

import jax.numpy as jnp

from hyprfine.parameters import cosmology
from hyprfine.utils.cosmology import n_H_tot


def test_n_H_tot_positive(cosmo: cosmology) -> None:
    """Total hydrogen density should be positive."""
    assert n_H_tot(jnp.array(50.0), cosmo) > 0


def test_n_H_tot_scales_as_cubed_redshift(cosmo: cosmology) -> None:
    """Hydrogen density should scale as (1+z)^3."""
    z1, z2 = jnp.array(10.0), jnp.array(20.0)
    ratio = n_H_tot(z2, cosmo) / n_H_tot(z1, cosmo)
    expected = ((1 + z2) / (1 + z1)) ** 3
    assert jnp.isclose(ratio, expected, rtol=1e-5)


def test_n_H_tot_increases_with_redshift(cosmo: cosmology) -> None:
    """Hydrogen density should be higher at higher redshift."""
    assert n_H_tot(jnp.array(100.0), cosmo) > n_H_tot(jnp.array(10.0), cosmo)


def test_n_H_tot_depends_on_omega_b(cosmo: cosmology) -> None:
    """Higher Omega_b should give higher hydrogen density."""
    from hyprfine.parameters import cosmology

    cosmo_high_b = cosmology(
        H0=cosmo.H0,
        Omega_m=cosmo.Omega_m,
        Omega_b=cosmo.Omega_b * 2,
        Omega_c=cosmo.Omega_c,
        Y_He=cosmo.Y_He,
    )
    assert n_H_tot(jnp.array(50.0), cosmo_high_b) > n_H_tot(
        jnp.array(50.0), cosmo
    )
