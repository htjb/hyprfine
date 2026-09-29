# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip install ".[dev]"

# Lint and format
ruff check .
ruff check . --fix
ruff format .

# Run tests
pytest

# Build and serve docs locally
pip install ".[docs]"
mkdocs serve
```

Code style: line length 79, Google-style docstrings, type annotations required. Pre-commit hooks enforce ruff and isort.

## Architecture

**hyprfine** is a GPU-accelerated simulation toolkit for the cosmological 21-cm signal (Dark Ages / Cosmic Dawn era), built on JAX for JIT compilation and vectorisation.

### Signal generation pipeline (`hyprfine/analytic/main.py`)

`generate_signal(f_grid, cosmo, z_init)` is the top-level entry point:

1. **Recombination** — calls the external HYREC-2 C code (wrapped in `recombination/hyrec.py`) to compute ionisation fraction `xe(z)` and gas temperature `Tk(z)`.
2. **Coupling** — computes the Wouthuysen-Field + collisional coupling coefficient `xc(z)` (`analytic/coupling_coeffs.py`).
3. **Temperatures** — derives CMB temperature `Tcmb(z)` and spin temperature `Ts(Tk, Tcmb, xc)` (`analytic/temperatures.py`).
4. **21cm brightness** — evaluates `T21(z, Tk, Tcmb, Ts, xe, cosmo)` (`analytic/signal.py`).

JAX `vmap` is used to vectorise steps 2–4 over the redshift/frequency grid.

### Key modules

| Module | Purpose |
|--------|---------|
| `parameters.py` | `cosmology`, `astrophysics`, `constants` namedtuples; `const` singleton with physical constants |
| `recombination/hyrec.py` | Compiles and runs HYREC-2 C code; `install_hyrec`, `set_up_hyrec`, `call_hyrec` |
| `analytic/sfrd.py` | Star formation rate density: halo mass function, star formation efficiency, mass accretion |
| `analytic/wouthuysen_field.py` | Lyman-α/β photon rates, photon recycling fractions |
| `utils/cosmology.py` | Hydrogen number density, matter density, growth factor, σ(M) via CosmoPowerJAX |
| `matterpower.py` | Linear matter power spectrum via CosmoPowerJAX emulator |

### External dependency: HYREC-2

The `HYREC-2/` directory contains the C recombination code. `hyrec.py` handles compilation and subprocess execution. `set_up_hyrec` must be called before `call_hyrec` for each cosmology.

### Parameters

Cosmological parameters are passed around as the `cosmology` namedtuple: `(H0, Omega_m, Omega_b, Omega_c, Y_He)`.

### Emulator work (current branch: `hyrec-emulator`)

`astroemu-hyrec.py` trains neural network emulators (via the `astroemu` library) to replace the slow HYREC C-code calls. Trained emulator weights are expected to be stored under `hyprfine/data/`.
