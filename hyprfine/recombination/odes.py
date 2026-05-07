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

    Based on the fitting formula from Shull & van Steenberg (1985), as given
    in Furlanetto & Stoever (2010) and implemented in 21cmFAST.

    Args:
        xe: Ionization fraction.

    Returns:
        Fraction of X-ray energy that goes into heating.
    """
    return 0.9971 * (1 - (1 - xe**0.2663) ** 1.3163)


@jax.jit
def f_ion_SSvS(xe: float) -> float:
    """Fraction of X-ray energy that goes into ionization, as a function of xe.

    Based on the fitting formula from Shull & van Steenberg (1985), as given
    in Furlanetto & Stoever (2010) and implemented in 21cmFAST.

    Args:
        xe: Ionization fraction.

    Returns:
        Fraction of X-ray energy that goes into ionization.
    """
    return 0.3908 * (1 - xe**0.4092) ** 1.7592


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
def dxe_dz(
    z: float,
    Tk: float,
    xe: float,
    cosmo: cosmology,
    Gamma_X: float,
    niondot: float,
    nH_cm3: float,
    dtdz: float,
) -> float:
    """dx_e/dz including recombination and X-ray secondary ionization.

    Args:
        z: Redshift.
        Tk: Kinetic temperature in K.
        xe: Ionization fraction.
        cosmo: Cosmology object.
        Gamma_X: X-ray ionization rate per H atom [s^-1].
        niondot: Ionization rate per H atom [s^-1].
        nH_cm3: Hydrogen number density in cm^-3.
        dtdz: dt/dz in seconds per unit redshift.

    Returns:
        dxe/dz.
    """
    # Case B recombination coefficient
    alpha_B = 2.6e-13 * (Tk / 1e4) ** (-0.76)  # cm^3/s

    # clumping factor
    C_HII = jnp.maximum(1.0, 2.9 * ((1 + z) / 6) ** (-1.1))

    recomb = C_HII * alpha_B * nH_cm3 * xe**2
    xray_ion = Gamma_X * (1 - xe)
    uv_ion = niondot / nH_cm3

    dxe = dtdz * (-recomb + xray_ion + uv_ion)
    # Prevent xe from exceeding 1: clamp derivative to non-negative when xe >= 1
    return jnp.where(xe >= 1.0, jnp.maximum(0.0, dxe), dxe)


@jax.jit
def evolve_igm(
    z_start: float,
    z_end: float,
    Tk_init: float,
    xe_init: float,
    cosmo: cosmology,
    astro: astrophysics,
    N_zgrid: int = 100,
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
    def interp_jx(j_nu, z):
        return jnp.interp(z, z_grid, j_nu)

    vmapped_interp_jx = jax.vmap(interp_jx, in_axes=(0, None))

    @jax.jit
    def vector_field(
        z: float,
        state: tuple[float, float],
        args: tuple[cosmology, astrophysics],
    ) -> tuple[float, float]:
        Tk, xe = state
        cosmo, astro = args

        xe = jnp.clip(xe, 0.0, 1.0)  # Ensure xe stays in physical range

        # Interpolate J_X at current z
        jx = vmapped_interp_jx(jx_grid_T, z)

        dtdz = dt_dz(z, cosmo)
        nH_cm3 = n_H_tot(z, cosmo) * 1e-6
        f_heat = f_heat_SSvS(xe)
        f_ion = f_ion_SSvS(xe)

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
            dTk_dz(z, Tk, xe, cosmo, Q_X, nH_cm3, dtdz),
            dxe_dz(z, Tk, xe, cosmo, Gamma_X, niondot, nH_cm3, dtdz),
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
        y0=(Tk_init, xe_init),
        args=(cosmo, astro),
        saveat=diffrax.SaveAt(ts=z_out_grid),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        max_steps=10000,
        throw=False
    )

    z_out = solution.ts
    Tk_out = solution.ys[0]
    xe_out = solution.ys[1]

    return z_out, Tk_out, xe_out
