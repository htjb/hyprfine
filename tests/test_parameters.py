"""Tests for hyprfine.parameters."""

import pytest

from hyprfine.parameters import const, cosmology


def test_cosmology_fields():
    cosmo = cosmology(
        H0=67.32, Omega_m=0.3158, Omega_b=0.0494, Omega_c=0.2664, Y_He=0.2454
    )
    assert cosmo.H0 == 67.32
    assert cosmo.Omega_m == 0.3158
    assert cosmo.Omega_b == 0.0494
    assert cosmo.Omega_c == 0.2664
    assert cosmo.Y_He == 0.2454


def test_cosmology_wrong_args():
    with pytest.raises(TypeError):
        cosmology(H0=67.32)


def test_constants_positive():
    """Physical constants should all be positive."""
    for field in const._fields:
        assert getattr(const, field) > 0, f"const.{field} should be positive"
