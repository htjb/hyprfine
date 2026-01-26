"""Simulation parameters for hyperfine."""

from collections import namedtuple

cosmology = namedtuple(
    "cosmo",
    [
        "H0",
        "Omega_m",
        "Omega_b",
        "Omega_c",
        "Omega_bh2",
        "Omega_ch2",
        "z_init",
        "Y_He",
    ],
)

# speed of light, gravitational constant, hydrogen mass fraction,
# and proton mass
constants = namedtuple(
    "constants", ["c", "G", "m_p", "A10", "Tstar", "Tcmb0", "m_e", "k_b"]
)
# X_H is the hydrogen mass fraction i.e. 1 - Y_He
const = constants(
    c=2.99792458e8,
    G=6.67430e-11,  # c in m/s, G in m^3/(kg*s^2)
    m_p=1.67e-27,
    A10=2.85e-15,
    Tstar=0.06817,
    Tcmb0=2.725,
    m_e=9.109e-31,
    k_b=1.380649e-23,
)  # m_p in kg, A10 in s^-1, T* in K
