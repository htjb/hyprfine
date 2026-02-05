"""Approximate the SFRD as in Munoz 2023."""

import jax
import jax.numpy as jnp

from hyprfine.matterpower import matterpowerspec
from hyprfine.parameters import const, cosmology


def fstar(
    epsilon: jnp.ndarray,
    alpha_star: jnp.ndarray,
    beta_star: jnp.ndarray,
    M_pivot: jnp.ndarray,
    M_h: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the star formation efficiency fstar.

    Args:
        epsilon: Normalization of the star formation efficiency.
        alpha_star: Power-law index for low-mass halos.
        beta_star: Power-law index for high-mass halos.
        M_pivot: Turnover mass in solar masses.
        M_h: Halo mass in solar masses.

    Returns:
        fstar: Star formation efficiency.
    """
    print(alpha_star, beta_star, M_pivot)
    M_turn = 3.3e7  # from zeus21 code
    f_duty = jnp.exp(-M_turn / M_h)
    f_star = (2.0 * epsilon * f_duty) / (
        (M_h / M_pivot) ** (-alpha_star) + (M_h / M_pivot) ** (-beta_star)
    )
    return f_star


def dmh_dt(m_h: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the halo mass accretion rate.

    Args:
        m_h: Halo mass in solar masses.
        z: Redshift.

    Returns:
        dm_h/dt: Halo mass accretion rate in solar masses per year.
    """
    return m_h * (1 + z) ** 2.5


def sigma0(Mh: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Calculate the variance of the density field.

    At mass scale Mh and redshift z=0.

    Args:
        Mh: Halo mass in solar masses.
        cosmo: cosmology parameters.

    Returns:
        sigma0: Variance of the density field at mass scale Mh.
    """

    def integrand(
        k: jnp.ndarray, pk: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        window_func = (
            3 * (jnp.sin(R * k) - k * R * jnp.cos(R * k)) / (R * k) ** 3
        )
        return k**2 / (2 * jnp.pi**2) * pk * jnp.abs(window_func) ** 2

    R = (
        3
        * Mh
        / (
            4
            * jnp.pi
            * const.rhom
            * (cosmo.Omega_b + cosmo.Omega_c)
            * (cosmo.H0 / 100) ** 2
        )
    ) ** (1 / 3)  # in Mpc
    kmodes, power = matterpowerspec(cosmo, z=0)
    vmapped_integrand = jax.vmap(integrand, (0, 0, None))
    integrand_values = jnp.array(
        [vmapped_integrand(kmodes, power, r) for r in R]
    )
    sigma_squared = jnp.trapezoid(integrand_values, kmodes, axis=1)
    sigma = jnp.sqrt(sigma_squared)
    return sigma


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


def sigma(Mh: jnp.ndarray, cosmo: cosmology, z: jnp.ndarray) -> jnp.ndarray:
    """Calculate the variance of the density field at redshift z.

    Args:
        Mh: Halo mass in solar masses.
        cosmo: cosmology parameters.
        z: Redshift.

    Returns:
        sigma: Variance of the density field at mass scale Mh and redshift z.
    """
    sigma0_Mh = sigma0(Mh, cosmo)
    D_z = growth_factor(z, cosmo=cosmo)
    return sigma0_Mh * D_z


def dmstar_dt(
    m_h: jnp.ndarray,
    z: jnp.ndarray,
    epsilon: jnp.ndarray,
    alpha_star: jnp.ndarray,
    beta_star: jnp.ndarray,
    M_pivot: jnp.ndarray,
    cosmo: cosmology,
) -> jnp.ndarray:
    """Calculate the star formation rate.

    Args:
        m_h: Halo mass in solar masses.
        z: Redshift.
        epsilon: Normalization of the star formation efficiency.
        alpha_star: Power-law index for low-mass halos.
        beta_star: Power-law index for high-mass halos.
        M_pivot: Turnover mass in solar masses.
        cosmo: cosmology parameters.

    Returns:
        dm_star/dt: Star formation rate in solar masses per year.
    """
    f_star = fstar(epsilon, alpha_star, beta_star, M_pivot, m_h)
    f_b = cosmo.Omega_b / cosmo.Omega_m
    dm_h_dt = dmh_dt(m_h, z)
    return f_star * f_b * dm_h_dt


def dn_dmh(Mh: jnp.ndarray, cosmo: cosmology, z: jnp.ndarray) -> jnp.ndarray:
    """Sheth-Tormann halo mass function.

    Args:
        Mh: Halo mass in solar masses.
        cosmo: cosmology parameters.
        z: Redshift.

    Returns:
        dn/dMh: Halo mass function in number density per solar mass.
    """
    Pst = 0.3
    Ast = 0.3222 * jnp.sqrt(2 / jnp.pi)
    qst = 0.85
    delta_crit = 1.686
    sigma_val = sigma(Mh, cosmo, z)
    nu = jnp.sqrt(qst) * delta_crit / sigma_val
    fnu = -Ast * (1 + (nu ** (-2 * Pst))) * jnp.exp(-(nu**2) / 2)
    dsigma_dMh = jnp.gradient(sigma_val, Mh)
    return fnu * (const.rhom / Mh) * (dsigma_dMh / sigma_val)


def mean_sfrd(
    z: jnp.ndarray,
    Mh: jnp.ndarray,
    epsilon: jnp.ndarray,
    alpha_star: jnp.ndarray,
    beta_star: jnp.ndarray,
    M_pivot: jnp.ndarray,
    cosmo: cosmology,
) -> jnp.ndarray:
    """Calculate the star formation rate density (SFRD).

    Args:
        Mh: Halo mass in solar masses.
        z: Redshift.
        epsilon: Normalization of the star formation efficiency.
        alpha_star: Power-law index for low-mass halos.
        beta_star: Power-law index for high-mass halos.
        M_pivot: Turnover mass in solar masses.
        cosmo: cosmology parameters.

    Returns:
        SFRD: Star formation rate density in solar masses per year
            per cubic megaparsec.
    """
    dmstar_dt_val = dmstar_dt(
        Mh, z, epsilon, alpha_star, beta_star, M_pivot, cosmo
    )
    dn_dmh_val = dn_dmh(Mh, cosmo, z)
    return jnp.trapezoid(dmstar_dt_val * dn_dmh_val, Mh)
