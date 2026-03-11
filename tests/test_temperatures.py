"""Tests for hyprfine.analytic.temperatures."""

import jax.numpy as jnp

from hyprfine.analytic.temperatures import Tcmb, Ts
from hyprfine.parameters import const


def test_tcmb_today() -> None:
    """Tcmb at z=0 should equal the CMB temperature today."""
    assert jnp.isclose(Tcmb(0.0), const.Tcmb0)


def test_tcmb_scales_with_redshift() -> None:
    """Tcmb should scale as T0 * (1 + z)."""
    z = 100.0
    assert jnp.isclose(Tcmb(z), const.Tcmb0 * (1 + z))


def test_tcmb_positive() -> None:
    """Tcmb should be positive for all redshifts."""
    for z in [0.0, 10.0, 100.0, 1000.0]:
        assert Tcmb(z) > 0


def test_ts_uncoupled_equals_tcmb() -> None:
    """With xc=0 (no coupling) spin temperature should equal Tcmb."""
    T_gas = jnp.array(50.0)
    T_cmb = jnp.array(30.0)
    xc = jnp.array(0.0)
    assert jnp.isclose(Ts(T_gas, T_cmb, xc), T_cmb)


def test_ts_strongly_coupled_approaches_tgas() -> None:
    """With very large xc, spin temperature should approach Tgas."""
    T_gas = jnp.array(50.0)
    T_cmb = jnp.array(30.0)
    xc = jnp.array(1e10)
    assert jnp.isclose(Ts(T_gas, T_cmb, xc), T_gas, rtol=1e-4)


def test_ts_between_tgas_and_tcmb() -> None:
    """Spin temperature should lie between Tgas and Tcmb."""
    T_gas = jnp.array(50.0)
    T_cmb = jnp.array(200.0)
    xc = jnp.array(1.0)
    T_spin = Ts(T_gas, T_cmb, xc)
    assert T_gas <= T_spin <= T_cmb
