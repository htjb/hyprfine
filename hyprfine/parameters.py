"""Simulation parameters for hyperfine."""

from collections import namedtuple

cosmology = namedtuple(
    "cosmo",
    [
        "H0",  # Hubble constant in km/s/Mpc
        "Omega_m",  # Matter density parameter
        "Omega_b",  # Baryon density parameter
        "Omega_c",  # Cold dark matter density parameter
        "Y_He",  # Helium mass fraction
        "ns",  # scalar spectral index of primordial fluctuations
        "ln1010As",  # amplitude of the primordial power spectrum
    ],
)

astrophysics = namedtuple(
    "astro",
    [
        "epsilon",  # star formation efficiency
        "alpha_star",  # power-law index for star formation efficiency
        "beta_star",  # power-law index for star formation efficiency
        "M_pivot",  # pivot mass for star formation efficiency
    ],
)


constants = namedtuple(
    "constants",
    [
        "c",  # speed of light
        "G",  # gravitational constant
        "m_p",  # proton mass
        "A10",  # Einstein A coefficient for 21cm transition
        "Tstar",  # 21cm transition temperature in K
        "Tcmb0",  # CMB temperature today in K
        "m_e",  # electron mass
        "k_b",  # Boltzmann constant
        "Msun",  # Solar mass in kg
        "Mpc",  # Megaparsec in meters
        "yr",  # Year in seconds
        "lyman_alpha_freq",  # Ly-α frequency in Hz
        "lyman_beta_freq",  # Ly-β frequency in Hz
        "lyman_limit",  # Lyman limit frequency in Hz
    ],
)

const = constants(
    c=2.99792458e8,  # m/s
    G=6.67430e-11,  # G in m^3/(kg*s^2)
    m_p=1.67e-27,  # kg
    A10=2.85e-15,  # s^-1
    Tstar=0.06817,  # K
    Tcmb0=2.725,  # K
    m_e=9.109e-31,  # kg
    k_b=1.380649e-23,
    Msun=1.989e30,  # kg
    Mpc=3.086e22,  # m
    yr=3.154e7,  # s
    lyman_alpha_freq=2.466e15,  # Hz
    lyman_beta_freq=2.922e15,  # Hz lyman_limit=3.289e15, # Hz
    lyman_limit=3.289e15,  # Hz
)
