"""Approximate the SFRD as in Munoz 2023."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import cosmology, astrophysics
from hyprfine.utils.cosmology import growth_factor, rhom, sigma, sigma0

@jax.jit
def fstar(
    astro: astrophysics,
    M_h: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the star formation efficiency fstar.

    Args:
        astro: astrophysics parameters.
        M_h: Halo mass in solar masses.
        z: Redshift.

    Returns:
        fstar: Star formation efficiency.
    """
    M_turn = 3.3e7 * ((1 + z) / (21)) ** (-1.5)
    f_duty = jnp.exp(-M_turn / M_h)
    f_star = (2.0 * astro.epsilon * f_duty) / (
        (M_h / astro.M_pivot) ** (-astro.alpha_star)
        + (M_h / astro.M_pivot) ** (-astro.beta_star)
    )
    return f_star

@jax.jit
def dmh_dt(M_z: jnp.ndarray, z: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Calculate the halo mass accretion rate.

    Approximation from Correa et al. 2015 (1409.5228), 
    which is more accurate than Fakhouri
    et al. 2010 and valid for a wider range of cosmologies.

    Args:
        M_z: Halo mass in solar masses at redshift z.
        z: Redshift.
        cosmo: cosmology parameters.
    
    Returns:
        dm_h/dt: Halo mass accretion rate in solar masses per year.
    """
    # Step 1: get z_f and q from M0 (eqs B2, B3)
    log10_M_z = jnp.log10(M_z)
    z_f = -0.0064 * log10_M_z**2 + 0.0237 * log10_M_z + 1.8837
    q = 4.137 * z_f**(-0.9476)

    # Step 2: f(M0) from sigma (eq B4)
    R0 = (3 * M_z / (4 * jnp.pi * rhom(0, cosmo))) ** (1/3)
    Rq = (3 * (M_z / q) / (4 * jnp.pi * rhom(0, cosmo))) ** (1/3)
    S0 = sigma0(R0, cosmo)**2
    Sq = sigma0(Rq, cosmo)**2
    f = 1.0 / jnp.sqrt(Sq - S0)

    # Step 3: a from growth factor derivative at z=0 (eq B6)
    dDdz_0 = jax.grad(lambda z: growth_factor(z, cosmo))(0.0)
    a = 1.686 * jnp.sqrt(2.0 / jnp.pi) * dDdz_0 + 1.0

    # Step 4: dM/dt (Correa+2015 eq on p.4)
    # M0 here is the halo mass at redshift z (used directly as M(z) in the
    # formula). The M(z) evolution step is skipped: the formula requires M(z)
    # evaluated at the observed redshift, which is just M0 itself.
    h = cosmo.H0 / 100.0
    E_z = jnp.sqrt(cosmo.Omega_m * (1 + z)**3 + (1 - cosmo.Omega_m))

    return (71.6 * (M_z / 1e12) * (h / 0.7)
            * f * ((1 + z) - a) * E_z)


@jax.jit
def dmstar_dt(
    m_h: jnp.ndarray,
    z: jnp.ndarray,
    astro: astrophysics,
    cosmo: cosmology,
) -> jnp.ndarray:
    """Calculate the star formation rate.

    Args:
        m_h: Halo mass in solar masses.
        z: Redshift.
        astro: astrophysics parameters.
        cosmo: cosmology parameters.

    Returns:
        dm_star/dt: Star formation rate in solar masses per year.
    """
    f_star = fstar(astro, m_h, z)
    f_b = cosmo.Omega_b / cosmo.Omega_m
    dm_h_dt = dmh_dt(m_h, z, cosmo)
    return f_star * f_b * dm_h_dt

@jax.jit
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
    fnu = Ast * nu * (1 + (nu ** (-2 * Pst))) * jnp.exp(-(nu**2) / 2)

    # Compute d(ln sigma)/d(ln M) numerically
    dln_sigma_dln_M = jnp.gradient(jnp.log(sigma_val), jnp.log(Mh))

    rhomatter = rhom(0, cosmo)
    return fnu * (rhomatter / Mh**2) * jnp.abs(dln_sigma_dln_M)

@jax.jit
def mean_sfrd(
    z: jnp.ndarray,
    Mh: jnp.ndarray,
    astro: astrophysics,
    cosmo: cosmology,
) -> jnp.ndarray:
    """Calculate the star formation rate density (SFRD).

    Args:
        Mh: Halo mass in solar masses.
        z: Redshift.
        astro: astrophysics parameters.
        cosmo: cosmology parameters.

    Returns:
        SFRD: Star formation rate density in solar masses per year
            per cubic megaparsec.
    """
    dmstar_dt_val = dmstar_dt(
        Mh, z, astro, cosmo
    )  # in solar masses per year !need to check??
    dn_dmh_val = dn_dmh(
        Mh, cosmo, z
    )  # in number density per solar mass per Mpc^3
    return jnp.trapezoid(dmstar_dt_val * dn_dmh_val * Mh, x=jnp.log(Mh))


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
    astro: astrophysics,
    cosmo: cosmology,
    z: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the star formation rate density (SFRD) at smoothing scale R.

    Args:
        R: Smoothing scale in Mpc.
        Mmin: Minimum halo mass in solar masses.
        Mmax: Maximum halo mass in solar masses.
        astro: astrophysics parameters.
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
        10 ** jnp.linspace(jnp.log10(Mmin), jnp.log10(Mmax), 100),
        astro,
        cosmo,
    )
    gamma_r = 0.5
    return mean_sfrd_val * jnp.exp(
        gamma_r * deltar - 0.5 * gamma_r**2 * sigma_R**2
    )
