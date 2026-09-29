# Bug Tracker

## Open

- **2026-06-26** `benchmark.py` line 180: `if gpus:` raises `NameError` when `--no-gpu` flag is passed, because `gpus` is only assigned inside the `try` block when `skip_gpu=False`. Pre-existing bug, not introduced by zeus21 addition.

## Recently Fixed (this session)

- **2026-06-26** `J_X` in `xrays.py`: `(1+z)²` at the observer was outside the integral; should be `(1+z')²` at the source shell inside the integral. Underestimated J_X by ~3× → Tk ~40% too low, xe undershooting then overshooting zeus21 via α_B cascade. Fixed.

## Fixed

- **2026-06-26** `dxe_dz` in `odes.py`: used quadratic recombination `xe²` for the UV term (homogeneous IGM model), while zeus21 uses a bubble model where recombination in HII regions scales as `Q` (linear). Equilibrium was `xe_eq ≈ 0.09` (hyprfine) vs `Q_eq ≈ 0.008` (zeus21) — the 10× discrepancy at z=10. Fixed by splitting into `xe_bg` (X-ray + HYREC, quadratic) and `Q` (UV bubbles, linear) as separate ODE state variables.
- **2026-06-26** `calculate_epsilon_x_intrinsic` in `xrays.py`: SED normalized over `d(nu_keV)` [keV] but then divided by `nu` [Hz] instead of `conv.keV_to_Hz` [Hz/keV]. This shifted the effective spectral index from `alpha_x` to `alpha_x - 1`, producing 2× too many soft (0.5 keV) X-ray photons → too much photoionization and heating → xe and Tk too high. Fixed by normalizing over `dν` (Hz) directly.
- **2026-06-26** `calculate_epsilon_alpha_tot` in `wouthuysen_field.py`: wrong `z_max_n` formula (always trivially true, so all Ly-n recycling fractions counted at all redshifts) and SED evaluated at wrong frequencies → J_alpha ~10× too large. Fixed.
- **2026-06-26** `Mmin=1e8` in `J_alpha`/`J_X` misses dominant halo contribution at z > 20. Fixed to `1e6`.
- **2026-06-26** `qst = 0.85` in `sfrd.py:dn_dmh` should be `0.707`; the Sheth-Tormen parameter is √0.5, not the Press-Schechter value. Fixed.
- **2026-06-26** `sigma0` used CosmoPowerJAX, whose k_max = 9.8 h/Mpc truncates the integral for M < 5×10⁹ M☉, causing sigma to plateau. Fixed by switching to EH98 analytic transfer function in `matterpower.py`. Three sub-bugs in the EH98 implementation were also fixed: wrong R(z) prefactor (31.5e3 → 31.5), wrong k unit conversion (k*h → k/h), missing ΛCDM growth suppression factor.
- **2026-06-22** `mkdocs.yaml` nav entry for the Dark Ages pointed to
  `the-dark_ages.md` (underscore) but the file is `the-dark-ages.md` (hyphen).
  Fixed.

## Won't Fix

*(none)*
