"""Cosmological helpers for hyperfine."""

import jax
import jax.numpy as jnp

from hyprfine.matterpower import matterpowerspec
from hyprfine.parameters import const, cosmology

@jax.jit
def chi_single(z_s: float, z: float, cosmo: cosmology) -> jnp.ndarray:
    """Calculate the comoving distance chi(z_s) from redshift z to z_s.

    Args:
        z_s: Source redshift.
        z: Observation redshift.
        cosmo: Cosmology object.

    Returns:
        chi: Comoving distance from z to z_s in Mpc.
    """
    z_int = jnp.linspace(z, z_s, 500)
    integrand = (const.c / 1e3) / H(z_int, cosmo) # Gives Mpc
    return jnp.trapezoid(integrand, z_int)

@jax.jit
def H(z: float, cosmo: cosmology) -> float:
    """Calculate the Hubble parameter H in s^-1.

    Args:
        z: Redshift.
        cosmo: Cosmology object.

    Returns:
        H(z): Hubble parameter at redshift z in s^-1.
    """
    Omega_L = 1.0 - cosmo.Omega_m
    return cosmo.H0 * jnp.sqrt(
        cosmo.Omega_m * (1 + z) ** 3 + Omega_L + cosmo.Omega_r * (1 + z) ** 4
    )


@jax.jit
def n_H_tot(z: int, cosmo: cosmology) -> jnp.ndarray:
    """Mean hydrogen number density in m^-3.

    Args:
        z: Redshift.
        cosmo: Cosmology object.

    Returns:
        n_H: Mean hydrogen number density in m^-3.
    """
    H0 = cosmo.H0 * 1e3 / const.Mpc  # Convert H0 from km/s/Mpc to s^-1
    rho_bar = (
        (3 * H0**2 / (8 * jnp.pi * const.G)) * cosmo.Omega_b * (1 + z) ** 3
    )
    return ((1 - cosmo.Y_He) * rho_bar) / const.m_p

@jax.jit
def rhom(z: int, cosmo: cosmology) -> jnp.ndarray:
    """Mean matter density in M_sun/Mpc^3.

    Args:
        z: Redshift.
        cosmo: Cosmology object.

    Returns:
        rhom: Mean matter density in M_sun/Mpc^3.
    """
    H0 = cosmo.H0 * 1e3 / const.Mpc

    rho_bar = (
        (3 * H0**2 / (8 * jnp.pi * const.G))
        * (cosmo.Omega_b + cosmo.Omega_c)
        * (1 + z) ** 3
    )  # in kg/m^3

    conversion_factor = const.Mpc**3 / const.Msun

    return rho_bar * conversion_factor  # in M_sun/Mpc^3

@jax.jit
def growth_factor(z: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Linear growth factor D(z), normalized to D(0)=1.

    Valid for flat LCDM.

    Args:
        z: Redshift.
        cosmo: cosmology parameters.

    Returns:
        D(z): Linear growth factor at redshift z.
    """
    Omega_m0 = cosmo.Omega_b + cosmo.Omega_c
    Omega_L0 = 1.0 - Omega_m0

    def E2(z: jnp.ndarray) -> jnp.ndarray:
        return Omega_m0 * (1 + z) ** 3 + Omega_L0

    def Omega_m(z: jnp.ndarray) -> jnp.ndarray:
        return Omega_m0 * (1 + z) ** 3 / E2(z)

    def Omega_L(z: jnp.ndarray) -> jnp.ndarray:
        return Omega_L0 / E2(z)

    def g(z: jnp.ndarray) -> jnp.ndarray:
        Om = Omega_m(z)
        Ol = Omega_L(z)
        return (5 * Om / 2) / (
            Om ** (4 / 7) - Ol + (1 + Om / 2) * (1 + Ol / 70)
        )

    return g(z) / (g(0.0) * (1 + z))


@jax.jit
def sigma0(R: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Calculate the variance of the density field.

    At smoothing scale R and redshift z=0.

    Args:
        R: Smoothing scale in Mpc.
        cosmo: cosmology parameters.

    Returns:
        sigma0: Variance of the density field at mass scale Mh.
    """

    def integrand(
        k: jnp.ndarray, pk: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        """Integrand for calculating sigma0.

        With k in h/Mpc, P(k) in (Mpc/h)^3, and R in Mpc/h,
        the product k*R is dimensionless and the full integrand
        k^2 * P(k) * W^2(kR) * dk is dimensionless, as required
        for sigma^2. No extra h factor is needed.

        Args:
            k: Wavenumber in h/Mpc.
            pk: Matter power spectrum in (Mpc/h)^3.
            R: Smoothing scale in Mpc/h
        """
        window_func = (
            3 * (jnp.sin(R * k) - k * R * jnp.cos(R * k)) / (R * k) ** 3
        )
        return k**2 / (2 * jnp.pi**2) * pk * window_func**2

    kmodes, power = matterpowerspec(cosmo, z=0)  # in 1/Mpc and Mpc^3
    R_h = R  # * cosmo.H0 / 100.0  # Convert R from Mpc to Mpc/h
    vmapped_integrand = jax.vmap(integrand, (None, None, 0))
    integrand_values = vmapped_integrand(kmodes, power, R_h)
    sigma_squared = jnp.trapezoid(integrand_values, kmodes, axis=1)
    sigma = jnp.sqrt(sigma_squared)
    return sigma

@jax.jit
def sigma(Mh: jnp.ndarray, cosmo: cosmology, z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the variance of the density field at redshift z.

    Args:
        Mh: Halo mass in solar masses.
        cosmo: cosmology parameters.
        z: Redshift.

    Returns:
        sigma: Variance of the density field at mass scale Mh and redshift z.
    """
    R = (
        3
        * Mh  # in M_sun
        / (
            4 * jnp.pi * rhom(0, cosmo)  # in M_sun/Mpc^3
        )
    ) ** (1 / 3)  # in Mpc

    sigma0_Mh = sigma0(R, cosmo)

    D_z = growth_factor(z, cosmo=cosmo)
    return sigma0_Mh * D_z
