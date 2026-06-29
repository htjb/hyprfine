"""Tests for hyprfine.parameters."""

import pytest  # type: ignore

from hyprfine.parameters import astrophysics, const, cosmology


def test_cosmology_fields() -> None:
    """Cosmology should have the expected fields with correct values."""
    cosmo = cosmology(
        H0=67.32,
        Omega_b=0.0494,
        Omega_c=0.2664,
        Y_He=0.2454,
        ns=0.965,
        ln1010As=3.044,
    )
    assert cosmo.H0 == 67.32
    assert cosmo.Omega_b == 0.0494
    assert cosmo.Omega_c == 0.2664
    assert cosmo.Y_He == 0.2454
    assert cosmo.ns == 0.965
    assert cosmo.ln1010As == 3.044


def test_cosmology_defaults() -> None:
    """Cosmology should be constructable with no arguments using defaults."""
    cosmo = cosmology()
    assert cosmo.H0 > 0
    assert cosmo.Omega_b > 0
    assert cosmo.Omega_c > 0


def test_cosmology_wrong_args() -> None:
    """Passing an unknown keyword to cosmology should raise a TypeError."""
    with pytest.raises(TypeError):
        cosmology(unknown_field=1.0)  # type: ignore


def test_astrophysics_fields() -> None:
    """Astrophysics should have the expected fields with correct defaults."""
    astro = astrophysics()
    assert astro.epsilon > 0
    assert astro.L40 > 0
    assert astro.N_alpha > 0
    assert astro.f_esc > 0
    assert astro.N_ion > 0


def test_constants_positive() -> None:
    """Physical constants should all be positive."""
    for field in const._fields:
        assert getattr(const, field) > 0, f"const.{field} should be positive"
