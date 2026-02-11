"""Approximate the SFRD as in Munoz 2023."""

import jax
import jax.numpy as jnp

from hyprfine.matterpower import matterpowerspec
from hyprfine.parameters import const, cosmology
from hyprfine.utils.cosmology import growth_factor, rhom


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
    M_turn = 3.3e7  # from zeus21 code in Msun
    f_duty = jnp.exp(-M_turn / M_h)
    f_star = (2.0 * epsilon * f_duty) / (
        (M_h / M_pivot) ** (-alpha_star) + (M_h / M_pivot) ** (-beta_star)
    )
    return f_star


def dmh_dt(M_h: jnp.ndarray, z: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Calculate the halo mass accretion rate.

    Args:
        M_h: Halo mass in solar masses.
        z: Redshift.
        cosmo: cosmology parameters.

    Returns:
        dm_h/dt: Halo mass accretion rate in solar masses per year.
    """
    # Convert H0 from km/s/Mpc to yr^-1:
    # H0 [km/s/Mpc] * 1e3 [m/km] / Mpc [m] * yr [s/yr] = yr^-1
    H0_per_year = cosmo.H0 * 1e3 / const.Mpc * const.yr
    A = 0.79 * H0_per_year * jnp.sqrt(cosmo.Omega_b + cosmo.Omega_c)
    return M_h * (1 + z) ** 2.5 * A


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
        return (
            k**2
            / (2 * jnp.pi**2)
            * pk
            * jnp.abs(window_func) ** 2
        )

    kmodes, power = matterpowerspec(cosmo, z=0)  # in h/Mpc and (Mpc/h)^3
    R_h = R * cosmo.H0 / 100.0  # Convert R from Mpc to Mpc/h
    vmapped_integrand = jax.vmap(integrand, (0, 0, None))
    integrand_values = jnp.array(
        [vmapped_integrand(kmodes, power, r) for r in R_h]
    )
    sigma_squared = jnp.trapezoid(integrand_values, kmodes, axis=1)
    sigma = jnp.sqrt(sigma_squared)
    return sigma


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
    dm_h_dt = dmh_dt(m_h, z, cosmo)
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
    qst = 0.707
    delta_crit = 1.686
    sigma_val = sigma(Mh, cosmo, z)
    nu = jnp.sqrt(qst) * delta_crit / sigma_val
    fnu = Ast * nu * (1 + (nu ** (-2 * Pst))) * jnp.exp(-(nu**2) / 2)

    # Compute d(ln sigma)/d(ln M) numerically
    dln_sigma_dln_M = jnp.gradient(jnp.log(sigma_val), jnp.log(Mh))

    rhomatter = rhom(z, cosmo)
    return fnu * (rhomatter / Mh) * jnp.abs(dln_sigma_dln_M)


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
    )  # in solar masses per year !need to check??
    dn_dmh_val = dn_dmh(
        Mh, cosmo, z
    )  # in number density per solar mass per Mpc^3
    return jnp.trapezoid(dmstar_dt_val * dn_dmh_val, Mh)


def delta_r(
    R: jnp.ndarray, cosmo: cosmology, z: jnp.ndarray, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the smoothed matter overdensity.

    Args:
        R: Smoothing scale in Mpc.
        cosmo: cosmology parameters.
        z: Redshift.
        key: JAX random key for generating the overdensity.

    Returns:
        delta_r: Critical overdensity for collapse.
    """
    sigma0_R = sigma0(R, cosmo)
    D_z = growth_factor(z, cosmo=cosmo)
    sigma_val = sigma0_R * D_z
    return jax.random.normal(key, shape=R.shape) * sigma_val, sigma_val


def sfrd(
    R: jnp.ndarray,
    Mmin: jnp.ndarray,
    Mmax: jnp.ndarray,
    epsilon: jnp.ndarray,
    alpha_star: jnp.ndarray,
    beta_star: jnp.ndarray,
    M_pivot: jnp.ndarray,
    cosmo: cosmology,
    z: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the star formation rate density (SFRD) at smoothing scale R.

    Args:
        R: Smoothing scale in Mpc.
        Mmin: Minimum halo mass in solar masses.
        Mmax: Maximum halo mass in solar masses.
        epsilon: Normalization of the star formation efficiency.
        alpha_star: Power-law index for low-mass halos.
        beta_star: Power-law index for high-mass halos.
        M_pivot: Turnover mass in solar masses.
        cosmo: cosmology parameters.
        z: Redshift.
        key: JAX random key for generating the overdensity.

    Returns:
        SFRD: Star formation rate density in solar masses per year
            per cubic megaparsec.
    """
    key, subkey = jax.random.split(key)
    deltar, sigma_R = delta_r(R, cosmo, z, subkey)
    mean_sfrd_val = mean_sfrd(
        z,
        10**jnp.linspace(jnp.log10(Mmin), jnp.log10(Mmax), 100),
        epsilon,
        alpha_star,
        beta_star,
        M_pivot,
        cosmo,
    )
    gamma_r = 0.5
    return mean_sfrd_val * jnp.exp(
        gamma_r * deltar - 0.5 * gamma_r**2 * sigma_R**2
    )
