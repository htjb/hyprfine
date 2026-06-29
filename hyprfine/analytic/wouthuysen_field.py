"""Lyman lines and Wouthuysen-Field coupling calculations."""

import jax
import jax.numpy as jnp

from hyprfine.analytic.sfrd import mean_sfrd
from hyprfine.parameters import astrophysics, const, conv, cosmology
from hyprfine.utils.cosmology import chi_single

# Build once at module level
_F_REC_VALUES = jnp.array([
    1.0,     # n=2
    0.0,     # n=3
    0.2609,  # n=4
    0.3078,  # n=5
    0.3259,  # n=6
    0.3353,  # n=7
    0.3410,  # n=8
    0.3448,  # n=9
    0.3476,  # n=10
    0.3496,  # n=11
    0.3512,  # n=12
    0.3512,  # n=13
    0.3535,  # n=14
    0.3543,  # n=15
    0.3550,  # n=16
    0.3556,  # n=17
    0.3561,  # n=18
    0.3565,  # n=19
    0.3569,  # n=20
    0.3572,  # n=21
    0.3575,  # n=22
    0.3578,  # n=23
])
_F_REC_DEFAULT = 0.358
_F_REC_N_MIN = 2

@jax.jit
def J_alpha(
    z: float,
    cosmo: cosmology,
    astro: astrophysics,
    Mmin: float = 1e6,
    Mmax: float = 1e16,
    N_shells: int = 50,
    z_max_source: float = 35.0,
) -> jnp.ndarray:
    """Calculate the Lyman-alpha flux J_alpha at redshift z.

    Implements Eq. 24 of Munoz et al. (2023):
        J_alpha(z) = (1+z)^2 / (4pi) * integral dR SFRD(R)
            epsilon_alpha^tot(nu')

    Args:
        z: Observation redshift.
        cosmo: Cosmology object.
        astro: Astrophysics object.
        Mmin: Minimum halo mass in solar masses.
        Mmax: Maximum halo mass in solar masses.
        N_shells: Number of radial shells.
        z_max_source: Maximum source redshift to integrate to.

    Returns:
        J_alpha: Lyman-alpha specific intensity at nu_Lalpha
            [cm^{-2} s^{-1} Hz^{-1} sr^{-1}].
    """
    # Build chi(z') table and invert to get z'(R)
    z_table = jnp.linspace(z + 0.01, z_max_source, 100)
    chi_table = jax.lax.map(lambda z_s: chi_single(z_s, z, cosmo), z_table)

    R = jnp.linspace(chi_table[0], chi_table[-1], N_shells)
    z_prime = jnp.interp(R, chi_table, z_table)

    # Halo mass grid
    Mh = 10 ** jnp.linspace(jnp.log10(Mmin), jnp.log10(Mmax), 50)

    # SFRD [M_sun/yr/Mpc³] and scalar emissivity at each shell
    sfrd_R = jax.lax.map(
        lambda z_p: mean_sfrd(z_p, Mh, astro, cosmo), z_prime
    )  # (N_shells,)
    eps_R = jax.lax.map(
        lambda zp: calculate_epsilon_alpha_tot(
            z_source=zp, z_21=z, astro=astro),
        z_prime,
    )  # (N_shells,)

    unit_factor = conv.Msun_to_kg / conv.yr_to_s / conv.Mpc_to_cm**2

    return (1 + z) ** 2 / (4 * jnp.pi) * jnp.trapezoid(
        sfrd_R * eps_R, R
    ) * unit_factor

@jax.jit
def calculate_epsilon_alpha_tot(
    z_source: float,  # redshift of the source
    z_21: float,  # redshift of the 21cm signal observation
    astro: astrophysics,
    n_max: int = 23,
) -> jnp.ndarray:
    """Calculate total epsilon_alpha^tot including recycling - Eq. 25.

    The total emissivity accounts for all Lyman transitions that
    can redshift into Ly-alpha at the observer's location.

    Args:
        z_source: Current source redshift
        z_21: Observer redshift
        astro: Astrophysics object (for SED parameters)
        n_max: Maximum Lyman level to consider

    Returns:
        Total effective emissivity at nu_Lalpha
            [same units as epsilon_alpha_intrinsic].
    """
    ns = jnp.arange(2, n_max + 1)  # shape (n_max - 1,)

    f_rec_n = jax.vmap(get_f_rec)(ns)

    # Frequency of each Lyman transition n -> 1
    nu_n = const.lyman_limit * (1.0 - 1.0 / ns**2)
    # Frequency at emission for each transition
    nu_n_prime = nu_n * (1.0 + z_source) / (1.0 + z_21)

    epsilon_intrinsic_n = calculate_epsilon_alpha_intrinsic(
        nu_n_prime, astro
    )

    ratio_n = (1.0 - 1.0 / (ns + 1.0)**2) / (1.0 - 1.0 / ns**2)
    z_max_n = (1.0 + z_21) * ratio_n - 1.0
    w_alpha_n = jnp.where(z_source < z_max_n, 1.0, 0.0)

    sum_eps = jnp.sum(f_rec_n * w_alpha_n * epsilon_intrinsic_n)

    return jnp.where(z_source < z_21, 0.0, sum_eps)


@jax.jit
def calculate_epsilon_alpha_intrinsic(
    nu: jnp.ndarray,
    astro: astrophysics,
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
        astro: Astrophysics object (for SED parameters)

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
        / (astro.alpha_low + 1)
        * (
            (nu_beta / nu_beta) ** (astro.alpha_low + 1)
            - (nu_alpha / nu_beta) ** (astro.alpha_low + 1)
        )
    )
    I_high = (
        nu_beta
        / (astro.alpha_high + 1)
        * (
            (nu_limit / nu_beta) ** (astro.alpha_high + 1)
            - (nu_beta / nu_beta) ** (astro.alpha_high + 1)
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
            A_low * (nu / nu_beta) ** astro.alpha_low,
            # Region 2: Ly-beta to Lyman limit (steep drop)
            A_high * (nu / nu_beta) ** astro.alpha_high,
        ),
    )

    return epsilon * astro.N_alpha / mu_b  # Scale to total photon number


def get_f_rec(n: int) -> jnp.ndarray:
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
    idx = n - _F_REC_N_MIN
    in_table = (idx >= 0) & (idx < len(_F_REC_VALUES))
    safe_idx = jnp.clip(idx, 0, len(_F_REC_VALUES) - 1)
    return jnp.where(in_table, _F_REC_VALUES[safe_idx], _F_REC_DEFAULT)


def get_lyman_freq(n: int) -> float:
    """Get Lyman series frequency for transition n -> 1.

    Args:
        n: Upper level (n >= 2)

    Returns:
        Frequency in Hz
    """
    return const.lyman_limit * (1 - 1 / n**2)
