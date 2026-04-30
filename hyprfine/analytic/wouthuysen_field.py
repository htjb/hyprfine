"""Lyman lines and Wouthuysen-Field coupling calculations."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import astrophysics, const, cosmology
from hyprfine.analytic.sfrd import mean_sfrd


def J_alpha(
    z: float,
    cosmo: cosmology,
    astro: astrophysics,
    Mmin: float = 1e8,
    Mmax: float = 1e16,
    N_shells: int = 200,
    z_max_source: float = 35.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the Lyman-alpha flux J_alpha at redshift z.

    Implements Eq. 24 of Munoz et al. (2023):
        J_alpha(z) = (1+z)^2 / (4pi) * integral dR SFRD(R) epsilon_alpha^tot(nu')

    Args:
        z: Observation redshift.
        cosmo: Cosmology object.
        astro: Astrophysics object.
        Mmin: Minimum halo mass in solar masses.
        Mmax: Maximum halo mass in solar masses.
        N_shells: Number of radial shells.
        z_max_source: Maximum source redshift to integrate to.

    Returns:
        nu: Frequency grid corresponding to epsilon_alpha^tot (shape (N_freq,))
        J_alpha: Lyman-alpha flux as a function of frequency, shape (N_freq,).
    """
    Omega_L = 1.0 - cosmo.Omega_m

    def H(zp):
        return cosmo.H0 * jnp.sqrt(cosmo.Omega_m * (1 + zp) ** 3 + Omega_L)

    def chi_single(z_s):
        z_int = jnp.linspace(z, z_s, 500)
        integrand = (const.c / 1e3) / H(z_int)  # km/s / (km/s/Mpc) = Mpc
        return jnp.trapezoid(integrand, z_int)

    # Build chi(z') table and invert to get z'(R)
    z_table = jnp.linspace(z + 0.01, z_max_source, 1000)
    chi_table = jax.vmap(chi_single)(z_table)

    R = jnp.linspace(chi_table[0], chi_table[-1], N_shells)
    z_prime = jnp.interp(R, chi_table, z_table)

    # Halo mass grid
    Mh = 10 ** jnp.linspace(jnp.log10(Mmin), jnp.log10(Mmax), 100)

    # SFRD and epsilon at each shell
    sfrd_R = jax.vmap(lambda zp: mean_sfrd(zp, Mh, astro, cosmo))(
        z_prime
    )  # (N_shells,)
    eps_R = jnp.array(
        [calculate_epsilon_alpha_tot(z_source=zp, z_21=z) for zp in z_prime]
    )  # (N_shells, N_freq)

    integrand = sfrd_R[:, None] * eps_R

    # same frequency grid as epsilon_alpha_tot
    nu = 10 ** jnp.arange(
        jnp.log10(const.lyman_alpha_freq), jnp.log10(const.lyman_limit), 0.001
    )

    # Unit conversions
    Mpc_to_cm = 3.086e24        # cm per Mpc
    yr_to_s = 3.154e7           # s per yr
    Msun_to_kg = 1.989e30       # kg per Msun

    # J_alpha has units M_sun yr^-1 Mpc^-2 kg^-1 from the integral
    # multiply by:
    #   Msun_to_kg   (M_sun -> kg, cancels with kg^-1 in epsilon)
    #   / yr_to_s    (yr^-1 -> s^-1)
    #   / Mpc_to_cm^2 (Mpc^-2 -> cm^-2)
    #   / (4*pi)     already divided, but need sr^-1 -- already there from 1/4pi
    #   the Hz^-1 comes from the fact that epsilon is per unit frequency implicitly

    unit_factor = Msun_to_kg / yr_to_s / Mpc_to_cm**2
    
    return nu, (1 + z) ** 2 / (4 * jnp.pi) * jnp.trapezoid(
        integrand, R, axis=0
    ) * unit_factor  # (N_freq,)


def calculate_epsilon_alpha_tot(
    z_source: float,  # redshift of the source (shell R where the photon was emitted)
    z_21: float,  # redshift of the 21cm signal observation
    n_max: int = 23,
    **sed_kwargs,
) -> jnp.ndarray:
    """Calculate total epsilon_alpha^tot including recycling - Eq. 25.

    The total emissivity accounts for all Lyman transitions that
    can redshift into Ly-alpha at the observer's location.

    Args:
        nu_prime: Redshifted frequency nu'[1 + z'(R)] / [1 + z]
        z: Current source redshift
        z_obs: Observer redshift
        n_max: Maximum Lyman level to consider
        N_alpha: Total photon number normalization
        **sed_kwargs: Arguments for epsilon_alpha_intrinsic

    Returns:
        Total effective emissivity
    """
    nu = 10 ** jnp.arange(
        jnp.log10(const.lyman_alpha_freq), jnp.log10(const.lyman_limit), 0.001
    )

    nu_prime = nu * (1 + z_source) / (1 + z_21)
    # Get intrinsic spectrum (unnormalized)
    epsilon_intrinsic = calculate_epsilon_alpha_intrinsic(
        nu_prime, **sed_kwargs
    )

    # Sum contributions from all Lyman transitions with recycling
    epsilon_tot = 0.0

    for n in range(2, n_max + 1):
        f_rec_n = get_f_rec(n)

        # w_alpha(n) is the "window function" - determines if this transition
        # can contribute at frequency nu_prime
        # It's 1 if nu_n can redshift to nu_prime
        # (i.e., if we're past z_max(n))
        # and 0 otherwise

        # Maximum redshift where nu_n redshifts to Ly-alpha
        z_max_n = (1 + z_source) * (1 - (1 + n) ** (-2)) / (1 - n ** (-2)) - 1

        # Window: contributes if z < z_max(n)
        # (photons from this transition have redshifted into range)
        w_alpha_n = jnp.where(z_source < z_max_n, 1.0, 0.0)

        epsilon_tot += f_rec_n * w_alpha_n * epsilon_intrinsic 

    return jnp.where(z_source < z_21, jnp.zeros_like(nu), epsilon_tot)


def calculate_epsilon_alpha_intrinsic(
    nu: jnp.ndarray,
    alpha_low: float = 0.14,
    alpha_high: float = -8.0,
    N_alpha: float = 9690.0,
) -> jnp.ndarray:
    """Calculate intrinsic stellar emissivity epsilon_alpha(nu) - Eq. 26.

    This is a double power law with a break at Ly-beta:
    - Below Ly-alpha: zero (can't produce Ly-alpha)
    - Ly-alpha to Ly-beta: epsilon propto
        nu^alpha_low (alpha_low = +0.14, rising)
    - Ly-beta to Lyman limit: epsilon propto
        nu^alpha_high (alpha_high = -8.0, steep drop)

    Args:
        nu: Frequency array [Hz]
        alpha_low: Power law index below Ly-beta (positive, spectrum rises)
        alpha_high: Power law index above Ly-beta (negative, steep cutoff)
        N_alpha: Total photon number normalization (arbitrary units)
            - defined such that integrating epsilon_alpha_intrinsic over
            the Ly-alpha to Lyman-limit band gives N_alpha photons.

    Returns:
        Intrinsic emissivity [arbitrary units, will be normalized]
    """
    # Normalized frequency
    nu_beta = const.lyman_beta_freq
    nu_alpha = const.lyman_alpha_freq
    nu_limit = const.lyman_limit
    # mean baryon mass in kg (assuming primordial composition)
    mu_b = 1.67352e-27  # kg

    # Unnormalised integrals of each power law segment
    # integral of (nu/nu_beta)^alpha from nu_a to
    # nu_b = nu_beta/(alpha+1) *
    #           [(nu_b/nu_beta)^(alpha+1) - (nu_a/nu_beta)^(alpha+1)]
    I_low = (
        nu_beta
        / (alpha_low + 1)
        * (
            (nu_beta / nu_beta) ** (alpha_low + 1)
            - (nu_alpha / nu_beta) ** (alpha_low + 1)
        )
    )
    I_high = (
        nu_beta
        / (alpha_high + 1)
        * (
            (nu_limit / nu_beta) ** (alpha_high + 1)
            - (nu_beta / nu_beta) ** (alpha_high + 1)
        )
    )

    # A_i set so A_low * I_low = 0.68 and A_high * I_high = 0.32
    A_low = 0.68 / I_low
    A_high = 0.32 / abs(
        I_high
    )  # abs because alpha_high+1 < 0 makes I_high negative

    # Three regions:
    epsilon = jnp.where(
        nu < nu_alpha,
        0.0,  # Below Ly-alpha: no contribution
        jnp.where(
            nu < nu_beta,
            # Region 1: Ly-alpha to Ly-beta (rising)
            A_low * (nu / nu_beta) ** alpha_low,
            # Region 2: Ly-beta to Lyman limit (steep drop)
            A_high * (nu / nu_beta) ** alpha_high,
        ),
    )

    return epsilon * N_alpha / mu_b  # Scale to total photon number


def get_f_rec(n: int) -> float:
    """Recycling fractions from Pritchard & Furlanetto (2006).

    What fraction of the non-direct decays eventually produce lyman
    alpha vs ending up in the 2s two photon decay.

    As defined in 21cmFAST (https://github.com/21cmfast/21cmFAST/
        blob/main/src/py21cmfast/src/heating_helper_progs.c#L200),
    in Zeus21 (https://github.com/JulianBMunoz/Zeus21/
        blob/main/zeus21/constants.py#L62)
    and in Echo21 (https://github.com/shikharmittal04/
        echo21/blob/master/src/echo21/const.py#L74)

    Args:
        n: Principal quantum number (n >= 2)

    Returns:
        Probability that cascade from level n produces Ly-alpha
    """
    f_rec_table = {
        2: 1.0,  # Ly-alpha
        3: 0.0,  # Ly-beta (usually absorbed locally)
        4: 0.2609,  # Ly-gamma
        5: 0.3078,
        6: 0.3259,
        7: 0.3353,
        8: 0.3410,
        9: 0.3448,
        10: 0.3476,
        11: 0.3496,
        12: 0.3512,
        13: 0.3512,
        14: 0.3535,
        15: 0.3543,
        16: 0.3550,
        17: 0.3556,
        18: 0.3561,
        19: 0.3565,
        20: 0.3569,
        21: 0.3572,
        22: 0.3575,
        23: 0.3578,
    }
    return f_rec_table.get(n, 0.358)


def get_lyman_freq(n: int) -> float:
    """Get Lyman series frequency for transition n -> 1.

    Args:
        n: Upper level (n >= 2)

    Returns:
        Frequency in Hz
    """
    return const.lyman_limit * (1 - 1 / n**2)
