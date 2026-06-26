"""ODEs for late time evolution of the IGM."""

import diffrax
import jax
import jax.numpy as jnp

from hyprfine.analytic.ionization import nion_dot
from hyprfine.analytic.xrays import _NU_X_GRID, J_X, sigma_X
from hyprfine.parameters import astrophysics, const, cosmology
from hyprfine.utils.cosmology import H, n_H_tot

@jax.jit
def f_heat_SSvS(xe: float) -> float:
    """Fraction of X-ray energy that goes into heating, as a function of xe.

    Fitting formula from Shull & van Steenberg (1985), Table 1 (300 eV
    primary electron). Implemented in 21cmFAST.

    Args:
        xe: Ionization fraction.

    Returns:
        Fraction of X-ray energy that goes into heating.
    """
    xe_safe = jnp.where(xe > 0, xe, 1e-10)  # Avoid zero to prevent NaNs
    return 0.9971 * (1 - (1 - xe_safe**0.2663) ** 1.3163)


@jax.jit
def f_ion_SSvS(xe: float) -> float:
    """Fraction of X-ray energy that goes into ionization, as a function of xe.

    Fitting formula from Shull & van Steenberg (1985), Table 1 (300 eV
    primary electron). Implemented in 21cmFAST.

    Args:
        xe: Ionization fraction.

    Returns:
        Fraction of X-ray energy that goes into ionization.
    """
    xe_safe = jnp.where(xe > 0, xe, 1e-10)  # Avoid zero to prevent NaNs
    return 0.3908 * (1 - xe_safe**0.4092) ** 1.7592


@jax.jit
def dt_dz(z: float, cosmo: cosmology) -> float:
    """dt/dz in seconds per unit redshift."""
    H_z = H(z, cosmo) * 1e3 / const.Mpc  # s^-1
    return -1.0 / (H_z * (1 + z))


@jax.jit
def dTk_dz(
    z: float,
    Tk: float,
    xe: float,
    cosmo: cosmology,
    Q_X: float,
    nH_cm3: float,
    dtdz: float,
) -> float:
    """dT_k/dz including adiabatic cooling, Compton heating, X-ray heating.

    Args:
        z: Redshift.
        Tk: Kinetic temperature in K.
        xe: Ionization fraction.
        cosmo: Cosmology object.
        Q_X: X-ray heating rate per unit volume [erg/s/cm^3].
        nH_cm3: Hydrogen number density in cm^-3.]
        dtdz: dt/dz in seconds per unit redshift.

    Returns:
        dTk/dz in K.
    """
    # Compton heating [K/s]
    T_cmb_z = const.Tcmb0 * (1 + z)
    u_cmb = (4 * const.sigma_SB_cgs / const.c_cgs) * T_cmb_z**4
    compton_rate = (
        (8 * const.sigma_T_cgs * u_cmb * xe)
        / (3 * const.m_e_cgs * const.c_cgs * (1 + xe + cosmo.Y_He / 4))
        * (T_cmb_z - Tk)
    )

    # X-ray heating [K/s]
    xray_rate = Q_X / (
        1.5 * const.k_b_cgs * (1 + xe + cosmo.Y_He / 4) * nH_cm3
    )

    return 2 * Tk / (1 + z) + dtdz * (compton_rate + xray_rate)


@jax.jit
def dxe_bg_dz(
    Tk: float,
    xe_bg: float,
    xe_total: float,
    Gamma_X: float,
    nH_cm3: float,
    dtdz: float,
) -> float:
    """dx_e_bg/dz for the background (HYREC residual + X-ray) ionization.

    Uses quadratic (homogeneous) recombination — appropriate for the diffuse
    IGM outside HII bubbles. X-rays ionize the remaining neutral fraction
    (1 - xe_total).

    Args:
        Tk: Kinetic temperature in K.
        xe_bg: Background ionization fraction (X-ray + HYREC residual).
        xe_total: Total mean ionization fraction (xe_bg + Q_HII).
        Gamma_X: X-ray secondary ionization rate per H atom [s^-1].
        nH_cm3: Hydrogen number density [cm^-3].
        dtdz: dt/dz [s].

    Returns:
        dxe_bg/dz.
    """
    alpha_B = 2.6e-13 * (jnp.maximum(Tk, 1.0) / 1e4) ** (-0.76)
    recomb = alpha_B * nH_cm3 * xe_bg**2
    xray_ion = Gamma_X * (1 - xe_total)
    return dtdz * (-recomb + xray_ion)


@jax.jit
def dQ_dz(
    z: float,
    Tk: float,
    Q: float,
    niondot: float,
    nH_cm3: float,
    dtdz: float,
) -> float:
    """dQ/dz for the UV HII bubble filling factor.

    Uses linear (bubble-model) recombination — recombinations occur only
    inside ionised regions (filling factor Q), so the rate is proportional
    to Q rather than Q^2.  This matches the zeus21 / Madau+1999 convention.

    Args:
        z: Redshift.
        Tk: Kinetic temperature in K.
        Q: HII bubble filling factor (UV contribution to mean xe).
        niondot: UV ionizing photon rate density [photons s^-1 cm^-3].
        nH_cm3: Hydrogen number density [cm^-3].
        dtdz: dt/dz [s].

    Returns:
        dQ/dz.
    """
    alpha_B = 2.6e-13 * (jnp.maximum(Tk, 1.0) / 1e4) ** (-0.76)
    C_HII = jnp.maximum(1.0, 2.9 * ((1 + z) / 6) ** (-1.1))
    recomb = C_HII * alpha_B * nH_cm3 * Q
    uv_ion = niondot / nH_cm3
    return dtdz * (uv_ion - recomb)


@jax.jit
def evolve_igm(
    z_start: float,
    z_end: float,
    Tk_init: float,
    xe_init: float,
    cosmo: cosmology,
    astro: astrophysics,
    N_zgrid: int = 50,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve T_k and x_e from z_start to z_end using diffrax.

    Integrates directly in redshift z following the approach of
    the original coupled_eq implementation.

    Args:
        z_start: Starting redshift (high z, e.g. 50).
        z_end: Ending redshift (low z, e.g. 6).
        Tk_init: Initial kinetic temperature in K.
        xe_init: Initial ionization fraction.
        cosmo: Cosmology object.
        astro: Astrophysics object.
        N_zgrid: Number of redshift points for J_X precomputation.

    Returns:
        z_out: Redshift array.
        Tk_out: Kinetic temperature array in K.
        xe_out: Ionization fraction array.
    """
    # Precompute J_X on a redshift grid
    z_grid = jnp.linspace(z_end, z_start, N_zgrid)
    nu = _NU_X_GRID
    jx_grid = jax.vmap(lambda z: J_X(z, cosmo, astro)[1])(z_grid)
    jx_grid_T = jx_grid.T

    # Precompute nion
    niondot_grid = jax.vmap(lambda z: nion_dot(z, cosmo, astro))(z_grid)

    # Precompute constants
    h_nu_HI = const.h_planck_cgs * 3.288e15  # erg
    sig = sigma_X(nu)

    @jax.jit
    def interp_jx(j_nu: jnp.ndarray, z: float) -> jnp.ndarray:
        return jnp.interp(z, z_grid, j_nu)

    vmapped_interp_jx = jax.vmap(interp_jx, in_axes=(0, None))

    @jax.jit
    def vector_field(
        z: float,
        state: tuple[float, float, float],
        args: tuple[cosmology, astrophysics, jnp.ndarray, jnp.ndarray],
    ) -> tuple[float, float, float]:
        Tk, xe_bg, Q = state
        cosmo, astro, jx_grid_T, niondot_grid = args

        xe_total = xe_bg + Q  # mean ionization fraction for T21 / Compton

        # Interpolate J_X at current z
        jx = vmapped_interp_jx(jx_grid_T, z)

        dtdz = dt_dz(z, cosmo)
        nH_cm3 = n_H_tot(z, cosmo) * 1e-6
        # SSvS fractions use background xe (secondaries in the neutral IGM)
        f_heat = f_heat_SSvS(xe_bg)
        f_ion = f_ion_SSvS(xe_bg)

        # X-ray heating rate per unit volume [erg/s/cm^3]
        Q_X = (
            4
            * jnp.pi
            * jnp.trapezoid(
                jx
                * nH_cm3
                * sig
                * f_heat
                * (1 - h_nu_HI / (const.h_planck_cgs * nu)),
                nu,
            )
        )

        # X-ray ionization rate per H atom [s^-1]
        Gamma_X = (
            4
            * jnp.pi
            * f_ion
            * jnp.trapezoid(jx * sig / (const.h_planck_cgs * nu), nu)
        )

        niondot = jnp.interp(z, z_grid, niondot_grid)

        return (
            dTk_dz(z, Tk, xe_total, cosmo, Q_X, nH_cm3, dtdz),
            dxe_bg_dz(Tk, xe_bg, xe_total, Gamma_X, nH_cm3, dtdz),
            dQ_dz(z, Tk, Q, niondot, nH_cm3, dtdz),
        )

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Kvaerno5()

    z_out_grid = jnp.linspace(z_start, z_end, 200)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=z_start,
        t1=z_end,
        dt0=-0.1,
        y0=(Tk_init, xe_init, 0.0),  # Q_HII starts at zero
        args=(cosmo, astro, jx_grid_T, niondot_grid),
        saveat=diffrax.SaveAt(ts=z_out_grid),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        max_steps=10000,
        throw=False
    )

    z_out = solution.ts
    Tk_out = solution.ys[0]
    xe_bg_out = solution.ys[1]
    Q_out = solution.ys[2]
    xe_out = xe_bg_out + Q_out  # total mean ionisation fraction

    return z_out, Tk_out, xe_out
