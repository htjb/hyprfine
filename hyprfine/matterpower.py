"""Matter power spectrum implementations.

Two implementations are provided:

``matterpowerspec``
    CosmoPowerJAX neural-network emulator (fast, JIT-compatible, but limited
    to k <= 9.8 h/Mpc — unreliable for M <= 5 x 10^{9} Modot).

``transfer_function_eh98`` / ``power_spectrum_eh98``
    Eisenstein & Hu (1998) analytic fitting formula, valid at all
    wavenumbers.  Used by ``sigma0`` in ``utils/cosmology.py``.

Reference
---------
Eisenstein, D. J. & Hu, W. 1998, ApJ, 496, 605
arXiv: astro-ph/9709066
"""

import jax
import jax.numpy as jnp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

from hyprfine.parameters import const, cosmology

_CPJ_MPK = CPJ(probe="mpk_lin")

# k grid for sigma0 integrals [Mpc^{-1}]; covers all halo scales from
# galaxy clusters (M ~ 10^{15} Modot)
# to the first star-forming halos (M ~ 10^{6} Modot).
_K_GRID = jnp.logspace(-4, 3, 2000)


@jax.jit
def matterpowerspec(
    cosmo: cosmology, z: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the matter power spectrum at redshift z via CosmoPowerJAX.

    Args:
        cosmo: Cosmology parameters.
        z: Redshift.

    Returns:
        Matter power spectrum at redshift z.
    """
    cosmo_params = {
        "omega_b": jnp.array([cosmo.Omega_b * (cosmo.H0 / 100) ** 2]),
        "omega_cdm": jnp.array([cosmo.Omega_c * (cosmo.H0 / 100) ** 2]),
        "h": jnp.array([cosmo.H0 / 100]),
        "n_s": jnp.array([cosmo.ns]),
        "ln10^{10}A_s": jnp.array([cosmo.ln1010As]),
        "z": jnp.array([z]),
    }
    emulator_predictions = _CPJ_MPK.predict(cosmo_params)
    return _CPJ_MPK.modes, emulator_predictions # type: ignore


@jax.jit
def transfer_function_eh98(k: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Eisenstein & Hu (1998) 'no-wiggle' CDM transfer function.

    Smooth (no-BAO) approximation from EH98 Appendix, accurate to ~5% for
    standard ΛCDM and valid at all wavenumbers.  Sufficient for computing
    sigma(M) and the halo mass function at cosmic-dawn scales.

    Args:
        k: Wavenumber in Mpc^{-1}.
        cosmo: Cosmological parameters.

    Returns:
        T(k): Dimensionless transfer function, shape matching k.

    Reference:
        Eisenstein & Hu (1998), ApJ 496, 605,
        Appendix (Eqs. 29-31).
    """
    h = cosmo.H0 / 100.0
    Theta = const.Tcmb0 / 2.7
    Omega_m = cosmo.Omega_b + cosmo.Omega_c
    omh2 = Omega_m * h**2
    ombh2 = cosmo.Omega_b * h**2
    fb = cosmo.Omega_b / Omega_m

    # Equality wavenumber [h/Mpc] (EH98 Eq. 3)
    k_eq = 7.46e-2 * omh2 * Theta**(-2)

    # Matter-radiation equality redshift (EH98 Eq. 2)
    z_eq = 2.5e4 * omh2 * Theta**(-4)

    # Baryon drag epoch (EH98 Eq. 4)
    b1 = 0.313 * omh2**(-0.419) * (1.0 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    z_drag = (
        1291.0
        * omh2**0.251
        / (1.0 + 0.659 * omh2**0.828)
        * (1.0 + b1 * ombh2**b2)
    )

    # Baryon-to-photon momentum ratio at equality / drag (EH98 Eqs. 5–6)
    R_eq = 31.5 * ombh2 * Theta**(-4) / (z_eq / 1e3)
    R_drag = 31.5 * ombh2 * Theta**(-4) / (z_drag / 1e3)

    # Sound horizon at drag epoch [Mpc/h] (EH98 Eq. 6)
    s = (
        (2.0 / (3.0 * k_eq))
        * jnp.sqrt(6.0 / R_eq)
        * jnp.log(
            (jnp.sqrt(1.0 + R_drag) + jnp.sqrt(R_drag + R_eq))
            / (1.0 + jnp.sqrt(R_eq))
        )
    )

    # No-wiggle shape parameter (EH98 Appendix Eq. 31)
    alpha_Gamma = (
        1.0
        - 0.328 * jnp.log(431.0 * omh2) * fb
        + 0.38 * jnp.log(22.3 * omh2) * fb**2
    )

    # EH98 fitting formulas use k in h/Mpc; input k is in Mpc^{-1}
    k_h = k / h
    Gamma_eff = Omega_m * h * (
        alpha_Gamma + (1.0 - alpha_Gamma) / (1.0 + (0.43 * k_h * s) ** 4)
    )

    # Dimensionless wavenumber (EH98 Appendix Eq. 30)
    q = k_h * Theta**2 / Gamma_eff

    # No-wiggle transfer function (EH98 Appendix Eq. 29)
    L0 = jnp.log(2.0 * jnp.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


@jax.jit
def power_spectrum_eh98(k: jnp.ndarray, cosmo: cosmology) -> jnp.ndarray:
    """Linear matter power spectrum at z = 0 using the EH98 transfer function.

    Normalization is derived from the primordial scalar amplitude A_s
    (Planck convention, pivot k_* = 0.05 Mpc^{-1}) via the sub-Hubble Poisson
    equation relating the gravitational potential to the density contrast.

    Args:
        k: Wavenumber in Mpc^{-1}.
        cosmo: Cosmological parameters.

    Returns:
        P(k): Linear matter power spectrum in Mpc³.

    Reference:
        Eisenstein & Hu (1998), ApJ 496, 605.
    """
    T = transfer_function_eh98(k, cosmo)
    Omega_m = cosmo.Omega_b + cosmo.Omega_c
    As = jnp.exp(cosmo.ln1010As) * 1e-10
    k_piv = 0.05  # Mpc⁻¹, Planck CMB pivot scale

    # Hubble length c/H0 in Mpc (H0 in km/s/Mpc, c in m/s → convert c to km/s)
    cH0 = (const.c / 1e3) / cosmo.H0  # Mpc

    # ΛCDM growth suppression: the Poisson-equation normalisation assumes
    # matter domination (D→a), but CLASS normalises to D_ΛCDM(z=0) which is
    # suppressed by dark energy.  Multiply by g²(Ω_m, Ω_Λ) — the Carroll+1992
    # growth function at z=0, the same formula used in utils/cosmology.py.
    Omega_L = 1.0 - Omega_m
    g0 = (5.0 * Omega_m / 2.0) / (
        Omega_m ** (4.0 / 7.0)
        - Omega_L
        + (1.0 + Omega_m / 2.0) * (1.0 + Omega_L / 70.0)
    )

    # P(k) = g₀² × (8π²/25) (c/H0)⁴ Ω_m⁻² A_s k (k/k_piv)^(n_s-1) T²(k)
    return (
        g0**2
        * (8.0 * jnp.pi**2 / 25.0)
        * cH0**4
        / Omega_m**2
        * As
        * k
        * (k / k_piv) ** (cosmo.ns - 1.0)
        * T**2
    )
