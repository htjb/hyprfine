"""Set up and call hyrec recombination code."""

import os
import subprocess

import numpy as np


def install_hyrec(clean: bool = False) -> None:
    """Code to install hyrec.

    Code downloads, compiles, and sets up hyrec.

    Args:
        clean: Whether to remove existing hyrec directory before installation.
    """
    if not os.path.exists("HYREC-2") or clean:
        subprocess.run(
            ["git", "clone", "https://github.com/nanoomlee/HYREC-2.git"],
            check=True,
        )
        subprocess.run(
            [
                "gcc",
                "-lm",
                "-O3",
                "hyrectools.c",
                "helium.c",
                "hydrogen.c",
                "history.c",
                "energy_injection.c",
                "hyrec.c",
                "-o",
                "hyrec",
            ],
            cwd="HYREC-2",
            check=True,
        )
    else:
        print("HYREC-2 directory already exists. Skipping installation.")


def set_up_hyrec(
    H0: np.ndarray,
    omb: np.ndarray,
    omc: np.ndarray,
    omk: np.ndarray,
    yhe: np.ndarray,
    base_dir: str = "./",
) -> None:
    """Code to set up hyrec.

    Code builds the input.dat file from a template
    given the cosmological parameters input to this class.

    Args:
        H0: Hubble parameter in km/s/Mpc.
        omb: Omega_baryon.
        omc: Omega_cdm.
        omk: Omega_curvature.
        yhe: Helium mass fraction.
        base_dir: Base directory to write the hyrec_input.dat file.
    """
    labels = [
        "h",
        "T0CMB",
        "Omega_b",
        "Omega_m",
        "Omega_k",
        "w0, wa",
        "Nmnu",
        "mnu1",
        "mnu2",
        "mnu3",
        "Y_He",
        "Neff",
        " ",
        "alpha(rec)/alpha(today)",
        "me(rec)/me(today)",
        "pann",
        "pann_halo",
        "ann_z",
        "ann_zmax",
        "ann_zmin",
        "ann_var",
        "ann_z_halo",
        "on_the_spot",
        "decay",
        " ",
        "Mpbh",
        "fpbh",
        " ",
        " ",
    ]

    with open("HYREC-2/input.dat") as file:
        data = {}
        for i, line in enumerate(file):
            if i <= 28:
                if labels[i] == " ":
                    data[i] = line.rstrip()
                else:
                    data[labels[i]] = line.rstrip()

    data["h"] = str(H0 / 100)
    data["Omega_b"] = str(omb)
    data["Omega_m"] = str(omc + omb)
    data["Omega_k"] = str(omk)
    data["Y_He"] = str(yhe)

    with open(base_dir + "hyrec_input.dat", "w") as file:
        for key, value in data.items():
            file.write(value + "\n")


def call_hyrec(
    base_dir: str = "./", redshift: int = 1100, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Code to call hyrec.

    Code runs the hyrec executable and returns the
    output as 3D grids with the same shape as the initial conditions.

    Args:
        base_dir: Base directory to find the hyrec_input.dat file.
        redshift: Redshift grid to interpolate the output to.
        verbose: Whether to let hyrec print to stdout.

    Returns:
        init_xe: Free electron fraction at the input redshift.
        init_Tk: Gas temperature at the input redshift.
    """
    # need to be in the hyrec dirtectory to run...
    os.chdir("HYREC-2")
    if verbose:
        os.system("./hyrec < ../" + base_dir + "/hyrec_input.dat")
    else:
        os.system(
            "./hyrec < ../"
            + base_dir
            + "/hyrec_input.dat"
            + "> /dev/null 2>&1"
        )
    os.chdir("..")

    data = np.loadtxt("HYREC-2/output_xe.dat")
    hyrec_z = data[:, 0][::-1]
    xe = data[:, 1][::-1]
    Tk = data[:, 2][::-1]

    init_xe = np.interp(redshift, hyrec_z, xe)
    init_Tk = np.interp(redshift, hyrec_z, Tk)  # in Kelvin

    return init_xe, init_Tk
