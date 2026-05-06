"""Code to calculate the X-ray SED and specific intensity J_x."""

import jax
import jax.numpy as jnp

from hyprfine.analytic.sfrd import mean_sfrd
from hyprfine.parameters import astrophysics, const, conv, cosmology
from hyprfine.utils.cosmology import H, chi_single, n_H_tot

vmapped_mean_sfrd = jax.vmap(mean_sfrd, in_axes=(0, None, None, None))
vmapped_chi_single = jax.vmap(chi_single, in_axes=(0, None, None))

_NU_X_GRID = jnp.logspace(
    jnp.log10(0.5 * conv.keV_to_Hz), jnp.log10(2.0 * conv.keV_to_Hz), 200
)


@jax.jit
def J_X(
    z: float,
    cosmo: cosmology,
    astro: astrophysics,
    Mmin: float = 1e8,
    Mmax: float = 1e13,
    N_shells: int = 200,
    z_max_source: float = 35.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the X-ray background intensity J_X at redshift z.

    Args:
        z: Observation redshift.
        cosmo: Cosmology object.
        astro: Astrophysics object.
        Mmin: Minimum halo mass in solar masses.
        Mmax: Maximum halo mass in solar masses.
        N_shells: Number of radial shells.
        z_max_source: Maximum source redshift to integrate to.

    Returns:
        nu: Frequency grid in Hz, shape (N_freq,)
        J_X: X-ray background intensity in erg/s/cm^2/Hz/sr, shape (N_freq,)
    """
    # Build chi(z') table and invert to get z'(R)
    z_table = jnp.linspace(z + 0.01, z_max_source, 1000)
    chi_table = vmapped_chi_single(z_table, z, cosmo)

    R = jnp.linspace(chi_table[0], chi_table[-1], N_shells)
    z_prime = jnp.interp(R, chi_table, z_table)

    # Halo mass grid
    Mh = 10 ** jnp.linspace(jnp.log10(Mmin), jnp.log10(Mmax), 100)

    # SFRD at each shell
    sfrd_R = vmapped_mean_sfrd(z_prime, Mh, astro, cosmo)  # shape (N_shells,)

    # X-ray emissivity at each shell (already includes attenuation)
    # eps_R = jnp.array(
    #     [
    #         calculate_epsilon_x_tot(
    #             z_source=zp, z_21=z, cosmo=cosmo, astro=astro
    #         )
    #         for zp in z_prime
    #     ]
    # )  # shape (N_shells, N_freq)
    eps_R = jax.vmap(
        lambda zp: calculate_epsilon_x_tot(
            z_source=zp, z_21=z, cosmo=cosmo, astro=astro
        )
    )(z_prime)  # shape (N_shells, N_freq)

    integrand = sfrd_R[:, None] * eps_R

    # Unit conversions: SFRD in Msun/yr/Mpc^3, epsilon in erg/s/Hz per Msun/yr
    # integral gives erg/s/Hz/Mpc^2, convert to erg/s/Hz/cm^2
    unit_factor = 1.0 / conv.Mpc_to_cm**2  # Mpc^-2 -> cm^-2

    # integral is over comoving shells, so we need to convert the SFRD from comoving to physical units
    # and the (1+z)^2 factor accounts for this

    return _NU_X_GRID, (1 + z) ** 2 / (4 * jnp.pi) * jnp.trapezoid(
        integrand, R, axis=0
    ) * unit_factor  # erg/s/cm^2/Hz/sr


@jax.jit
def calculate_epsilon_x_tot(
    z_source: float,
    z_21: float,
    cosmo: cosmology,
    astro: astrophysics,
) -> jnp.ndarray:
    """Total X-ray emissivity including IGM attenuation.

    Args:
        z_source: Source redshift
        z_21: Observer redshift
        cosmo: Cosmology object
        astro: Astrophysics object (for SED parameters)

    Returns:
        Attenuated X-ray emissivity, shape (N_freq,)
    """
    # Frequency at source
    nu_prime = _NU_X_GRID * (1 + z_source) / (1 + z_21)

    # Intrinsic emissivity at source frequency
    epsilon_intrinsic = calculate_epsilon_x_intrinsic(nu_prime, astro)

    # Attenuation along line of sight for each frequency
    tau = vmapped_tau_X(_NU_X_GRID, z_21, z_source, cosmo)
    # tau = jnp.zeros_like(nu)  # Placeholder: no attenuation for now

    return jnp.where(z_source > z_21, epsilon_intrinsic * jnp.exp(-tau), 0.0)


vmapped_calculate_epsilon_x_tot = jax.vmap(
    calculate_epsilon_x_tot, in_axes=(0, None, None, None)
)


@jax.jit
def calculate_epsilon_x_intrinsic(
    nu: jnp.ndarray, astro: astrophysics
) -> jnp.ndarray:
    """Calculate intrinsic X-ray emissivity epsilon_x(nu).

    This is a power law with index alpha_x, normalized such that the
    integrated luminosity in the 0.5-2 keV band is L40 * 1e40 erg/s per.

    Args:
        nu: Frequency array [Hz]
        astro: Astrophysics object (for SED parameters)

    Returns:
        Intrinsic X-ray emissivity ergs/s/SFR/Hz
    """
    nu_keV = nu / conv.keV_to_Hz  # Convert frequency from Hz to keV
    # Mask to 0.5-2 keV band
    in_band = (nu_keV >= astro.nu_0) & (nu_keV <= 2.0)

    # Unnormalized power law, only where in band
    Ix = jnp.where(in_band, nu_keV**astro.alpha_x, 0.0)

    # Normalize so integral over band = 1 (in keV)
    nu_keV_band = jnp.where(in_band, nu_keV, 0.0)
    norm = jnp.trapezoid(Ix, nu_keV_band)
    norm = jnp.where(norm == 0.0, 1.0, norm)  # Avoid division by zero
    Ix_normalized = jnp.where(in_band, Ix / norm, 0.0)

    log_epsilon_x = (
        jnp.log10(astro.L40)
        + 40.0
        + jnp.log10(
            jnp.where(
                Ix_normalized > 0, Ix_normalized, 1.0
            )  # Avoid log of zero
        )
        - jnp.log10(nu)
    )

    epsilon_x = jnp.where(
        in_band, 10**log_epsilon_x, 0.0
    )  # Set to 0 outside band

    return epsilon_x


@jax.jit
def sigma_X(nu: jnp.ndarray) -> jnp.ndarray:
    """HI photoionization cross section.

    Standard power law approximation, valid for X-ray energies.
    sigma_0 = 6.3e-18 cm^2 at nu_HI (13.6 eV threshold)

    Args:
        nu: Frequency in Hz
    Returns:
        Cross section in cm^2
    """
    nu_HI = 3.288e15  # Hz, HI ionization threshold (13.6 eV)
    sigma_0 = 6.3e-18  # cm^2
    return jnp.where(nu >= nu_HI, sigma_0 * (nu / nu_HI) ** (-3), 0.0)


@jax.jit
def tau_X(
    nu_obs: float,
    z_obs: float,
    z_source: float,
    cosmo: cosmology,
) -> float:
    """X-ray optical depth between z_obs and z_source.

    Args:
        nu_obs: Observed frequency in Hz
        z_obs: Observer redshift
        z_source: Source redshift
        cosmo: Cosmology object
    Returns:
        Optical depth (dimensionless)
    """
    z_int = jnp.linspace(z_obs, z_source, 200)

    # Frequency at each redshift along the path
    nu_z = nu_obs * (1 + z_int) / (1 + z_obs)

    # n_H in cm^-3 (convert from m^-3)
    nH = n_H_tot(z_int, cosmo) * 1e-6

    # Cross section in cm^2
    sig = sigma_X(nu_z)

    # integrand: n_H * sigma / H(z) / (1+z), integrated over dz
    integrand = nH * sig * (const.c * 1e2) / (H(z_int, cosmo) * (1 + z_int))

    return jnp.trapezoid(integrand, z_int)


vmapped_tau_X = jax.vmap(tau_X, in_axes=(0, None, None, None))
