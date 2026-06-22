"""Simulation parameters for hyperfine."""

from collections import namedtuple

cosmology = namedtuple(
    "cosmo",
    [
        "H0",  # Hubble constant in km/s/Mpc
        "Omega_b",  # Baryon density parameter
        "Omega_c",  # Cold dark matter density parameter
        "Y_He",  # Helium mass fraction
        "ns",  # scalar spectral index of primordial fluctuations
        "ln1010As",  # amplitude of the primordial power spectrum
    ],
    defaults=(67.36, 0.049, 0.266, 0.245, 0.97, 3.044),
)

astrophysics = namedtuple(
    "astro",
    [
        "epsilon",  # star formation efficiency
        "alpha_star",  # power-law index for star formation efficiency
        "beta_star",  # power-law index for star formation efficiency
        "M_pivot",  # pivot mass for star formation efficiency
        "L40", # X-ray luminosity per unit SFR in units of 10^40 erg/s/(Msun/yr)
        "alpha_x", # X-ray spectral index (negative for typical spectra)
        "nu_0", # Reference frequency for X-ray normalization (0.5 keV in keV)
        "alpha_low", # power law index for Lya emissivity below the Lyman limit
        "alpha_high", # power law index for Lya emissivity above the Lyman limit
        "N_alpha", # total number of Lyman-alpha photons produced per baryon in stars
        "f_esc", # escape fraction of ionizing photons
        "N_ion", # number of ionizing photons produced per baryon in stars
    ],
    defaults=(0.1, 0.5, -0.5, 3e11,
              1.0, -1.0, 0.5,
                0.14, -8.0, 9690.0,
              0.15, 5000.0),
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
        "sigma_SB_cgs",  # Stefan-Boltzmann constant in cgs units (erg/cm^2/s/K^4)
        "sigma_T_cgs",  # Thomson cross-section in cgs units (cm^2)
        "h_planck_cgs",  # Planck constant in erg·s
        "m_e_cgs",  # electron mass in g
        "c_cgs",  # speed of light in cm/s
        "k_b_cgs",  # Boltzmann constant in erg/K
    ],
)

conversion_factors = namedtuple(
    "conversion_factors",
    [
        "keV_to_Hz",  # Conversion factor from keV to Hz
        "Mpc_to_cm",  # Conversion factor from Mpc to cm
        "Msun_to_kg",  # Conversion factor from solar masses to kg
        "yr_to_s",  # Conversion factor from years to seconds
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
    sigma_T_cgs = 6.652e-25,  # cm^2, Thomson cross section
    sigma_SB_cgs = 5.671e-5,  # erg/cm^2/s/K^4, Stefan-Boltzmann constant
    h_planck_cgs=6.626e-27,      # erg·s
    m_e_cgs=9.109e-28,       # g
    c_cgs=2.998e10,          # cm/s
    k_b_cgs=1.381e-16,       # erg/K
)

conv = conversion_factors(
    keV_to_Hz=2.418e17,  # Hz/keV
    Mpc_to_cm=3.086e24,  # cm/Mpc
    Msun_to_kg=1.989e30,  # kg/Msun
    yr_to_s=3.154e7,  # s/yr
)
