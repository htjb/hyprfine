"""Lyman lines and Wouthuysen-Field coupling calculations."""

import jax
import jax.numpy as jnp

from hyprfine.parameters import const


def calculate_epsilon_alpha_tot(
    z: float,  # redshift of the source (shell R where the photon was emitted)
    z_source: float,  # redshift of the 21cm signal observation
    n_max: int = 23,
    N_alpha: float = 9690.0,
    **sed_kwargs,
) -> float:
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

    nu_prime = nu * (1 + z_source) / (1 + z)
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

        nu_n = get_lyman_freq(n)

        # Maximum redshift where nu_n redshifts to Ly-alpha
        # [1 + z_max(n)] / [1 + z] = nu_n / nu_alpha
        z_max_n = (1 + z_source) * (nu_n / const.lyman_alpha_freq) - 1
        if z == 20:
            print(f"n={n}, nu_n={nu_n:.2e} Hz, z_max(n)={z_max_n:.2f}")
        
        # Window: contributes if z < z_max(n)
        # (photons from this transition have redshifted into range)
        w_alpha_n = jnp.where(z < z_max_n, 1.0, 0.0)

        epsilon_tot += f_rec_n * w_alpha_n * epsilon_intrinsic

    # Normalize by N_alpha (total photons in Ly-alpha to Lyman-limit band)
    # This requires integrating epsilon_intrinsic, but in practice
    # N_alpha is defined to give the correct normalization
    epsilon_tot *= N_alpha

    return epsilon_tot


def calculate_epsilon_alpha_intrinsic(
    nu: jnp.ndarray,
    alpha_low: float = 0.14,
    alpha_high: float = -8.0,
    f_low: float = 0.68,  # Fraction of photons between Ly-alpha and Ly-beta
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
        f_low: Fraction of total photons between 
            Ly-alpha and Ly-beta (0.68 default)

    Returns:
        Intrinsic emissivity [arbitrary units, will be normalized]
    """
    # Normalized frequency
    nu_beta = const.lyman_beta_freq
    nu_alpha = const.lyman_alpha_freq

    # Three regions:
    epsilon = jnp.where(
        nu < nu_alpha,
        0.0,  # Below Ly-alpha: no contribution
        jnp.where(
            nu < nu_beta,
            # Region 1: Ly-alpha to Ly-beta (rising)
            (nu / nu_beta) ** alpha_low,
            # Region 2: Ly-beta to Lyman limit (steep drop)
            (nu / nu_beta) ** alpha_high,
        ),
    )

    return epsilon


def get_f_rec(n: int) -> float:
    """Recycling fractions from Pritchard & Furlanetto (2006).

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
        23: 0.3578
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
