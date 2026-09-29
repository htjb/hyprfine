# Changelog

## 2026-06-26 (6)

- Fixed `J_X` in `xrays.py`: `(1+z)²` factor used the observer redshift outside the integral, but the correct formula (Mesinger+2011/Furlanetto+2006) requires `(1+z')²` at each source shell inside the integral. For sources at z'~15–25 observed at z~10, this underestimated J_X by ~3×, causing Tk to be ~40% too low and xe to undershoot then overshoot zeus21 via the α_B cascade.

## 2026-06-26 (5)

- Added zeus21 CPU benchmark to `bin/benchmark.py`. Times one `get_T21_coefficients` call (no JIT compilation overhead). Shared parameter block at top ensures hyprfine and zeus21 use identical Planck 2018 cosmology and fiducial astrophysics (previously they differed on H0, ns, L40, f_esc). Zeus21 result shown as a single seagreen bar; zeus21 signal overlaid on the signal panel.

## 2026-06-26 (4)

- Fixed `xe` being ~10× too large at z=10 vs zeus21. Root cause: `dxe_dz` used
  quadratic recombination (`xe²`) for the UV reionisation term, matching a
  uniform-IGM model. The correct model (zeus21 / Madau+1999) tracks HII bubbles
  separately with linear recombination (`Q`). The ODE in `evolve_igm` now
  carries three state variables `(Tk, xe_bg, Q)`: `xe_bg` (HYREC residual +
  X-ray secondaries, quadratic recomb) and `Q` (UV HII bubble filling factor,
  linear/bubble recomb). Reported `xe = xe_bg + Q`. Old equilibrium
  `xe_eq = sqrt(nion/α_B/nH²) ≈ 0.09`; correct `Q_eq = nion/(C α_B nH²) ≈
  0.008`, consistent with zeus21.
- Corrected misleading docstrings in `f_heat_SSvS` / `f_ion_SSvS`: coefficients
  are from Shull & van Steenberg (1985) Table 1 (as used in 21cmFAST), not from
  Furlanetto & Stoever (2010) who use a different energy-dependent model.
- Renamed `dxe_dz` → `dxe_bg_dz` and added `dQ_dz` to reflect the split.

## 2026-06-26 (3)

- Fixed `calculate_epsilon_x_intrinsic` in `xrays.py`: the SED was normalized over `d(nu_keV)` (keV) but then divided by `nu` (Hz) instead of the constant `conv.keV_to_Hz`. This gave an effective spectral index of `alpha_x − 1` instead of `alpha_x`, producing a spectrum that was too soft and causing xe and Tk to be too high at z < 20. Emissivity now normalised over `dν` (Hz) directly so `∫ ε dν = L40 × 1e40` exactly.

## 2026-06-26 (2)

- Fixed `calculate_epsilon_alpha_tot` in `wouthuysen_field.py`: (1) `z_max_n` formula was wrong — it depended on `z_source` and was always trivially satisfied, so all Ly-n recycling fractions were always included regardless of whether they were in the photon window. Correct formula: `z_max_n = (1+z_21) × ν_L(n+1)/ν_Ln − 1`. (2) The SED was being evaluated over the full frequency grid instead of at each Ly-n line frequency. Both bugs caused J_alpha to be ~10× too large. Fix was contributed by Gemini; this commit also refactors `J_alpha` and `x_alpha` to return scalars now that the frequency grid is no longer needed.
- Fixed minimum halo mass in `J_alpha`, `J_X`, and SFRD integrals: was `1e8 M_sun`, which misses the dominant contribution at z > 20 where relevant halos are M ~ few × M_turn ~ 10^7 M_sun. Changed to `1e6 M_sun`.
- Changed HYREC → ODE transition redshift from z=50 to z=35.

## 2026-06-26

- Fixed Sheth-Tormen parameter `qst = 0.85 → 0.707` in `sfrd.py:dn_dmh` (was using the Press-Schechter value; ST requires sqrt(0.5) ≈ 0.707).
- Replaced CosmoPowerJAX emulator in `sigma0` with Eisenstein & Hu (1998) no-wiggle analytic transfer function, fixing sigma plateau at M < 5×10⁹ M☉ caused by the emulator's k_max truncation at 9.8 h/Mpc.
  - Added `transfer_function_eh98` and `power_spectrum_eh98` to `matterpower.py`.
  - Three bugs fixed during implementation: `31.5e3 → 31.5` in the baryon-to-photon momentum ratio R(z); `k_h = k*h → k/h` (EH98 uses k in h/Mpc, input is Mpc⁻¹); added ΛCDM growth suppression factor g(Ω_m)² to match CLASS normalization convention.

## 2026-06-22 (10)

- Reverted `lax.map` → `vmap` for the J_X and nion_dot precomputation loops inside `evolve_igm`. Now that the shell loops inside J_X are `lax.map`, the nested explosion is gone and the peak memory for vmapping over z_grid is only B × N_zgrid × (tau_X step) ≈ B × 8 MB — safe at B=100 on a T4. Restores GPU parallelism over the N_zgrid=50 precomputation points.

## 2026-06-22 (9)

- Reduced integration grid sizes to cut memory when batching: `z_table` 200→100 (chi lookup in J_X/J_alpha), `z_int` 200→100 (tau_X optical depth), `Mh` 100→50 (halo mass grid in J_X, J_alpha, sfrd), `N_zgrid` default 100→50 (J_X precomputation grid in evolve_igm).
- Also converted the two remaining `jax.vmap` calls inside `evolve_igm` (J_X precomputation and nion_dot precomputation over z_grid) to `jax.lax.map` — this was the main remaining bottleneck, as B outer-batched cosmologies × N_zgrid simultaneous J_X calls still caused OOMs even after the earlier lax.map changes.

## 2026-06-22 (8)

- Updated `bin/benchmark.py` to vmap Cosmic Dawn signals as well as Dark Ages now that internal `lax.map` changes make it memory-safe. Single `batched_generate` handles both modes; plot shows DA (solid circles) and CD (dashed squares) on shared total-time and ms/signal panels.

## 2026-06-22 (7)

- Replaced `jax.vmap` with `jax.lax.map` for the three internal loops inside `J_X` (`xrays.py`) and `J_alpha` (`wouthuysen_field.py`): the chi-table build (200 points), the per-shell SFRD (30 shells), and the per-shell emissivity (30 shells). This eliminates the B × 100 × 200 × 30 nested tensor blowup that caused OOM at batch size 10 when vmapping `generate_signal` over cosmologies. `vmapped_tau_X` (200 frequencies) inside `calculate_epsilon_x_tot` keeps its vmap — at that level it's fine.

## 2026-06-22 (6)

- Rewrote `bin/benchmark.py`: separates dark ages (vmapped batch sweep over cosmologies) from cosmic dawn (single-signal timing only), after discovering the nested vmaps inside the ODE solver OOM even at batch=10. Three-panel plot: single-signal bar chart, batched total time, batched ms/signal.

## 2026-06-22 (5)

- Refactored existing tests to match the current API: removed `Omega_m` from `cosmology` fixture, fixed `kappa(Tk, xe)` signature (was 3 args, now 2), fixed `Ts(T_gas, T_cmb, xc, xalpha)` signature (was 3 args, now 4), fixed `T21(z, T_cmb, T_s, xe, cosmo)` signature (old `T_gas` arg removed), fixed `test_cosmology_wrong_args` (all fields now have defaults so passing `H0` alone is valid).
- Added Cosmic Dawn tests: `test_sfrd.py` (fstar, HMF, SFRD), `test_wouthuysen.py` (recycling fractions, emissivity), `test_xrays.py` (sigma_X, tau_X, epsilon_x), and Cosmic Dawn coverage in `test_signal.py` (shape, finiteness, xe bounds, Tk positivity).
- Added `astro` fixture to `conftest.py`.
- All 55 tests pass.

## 2026-06-22 (4)

- Added `docs/tutorials/cosmic-dawn.md`: tutorial covering the frequency grid, `astrophysics` parameter table, calling `generate_signal` with astrophysics, plotting output, and varying astrophysical parameters.

## 2026-06-22 (3)

- Minor prose edits to `wouthuysen.md` and `xrays.md` to add introductory context sentences describing the physical role of Lyman-α and X-ray emission in the 21-cm signal.

## 2026-06-22 (2)

- Fixed broken cross-links in `the-dark-ages.md` and `recombination/hyrec_emulators.md`: removed incorrect `docs/modelling/` prefixes and added missing `../` for links from the `recombination/` subdirectory to parent-level pages.

## 2026-06-22

- Added `docs/modelling/recombination/cosmic-dawn.md`: documents the coupled ODE
  system for $T_k$ and $x_e$ evolution from $z=50$ through reionisation,
  covering adiabatic cooling, Compton heating, X-ray heating/ionisation, and UV
  photoionisation.
- Added `docs/modelling/wouthuysen.md`: documents the Wouthuysen-Field effect,
  the Lyman-$\alpha$ flux integral, the stellar SED double power law model,
  photon recycling fractions, and the SFRD model (Sheth-Tormen HMF + Correa
  et al. accretion + double power law star formation efficiency).
- Added `docs/modelling/xrays.md`: documents the X-ray background intensity
  integral, the power-law SED model, IGM attenuation via the HI photoionisation
  cross section, and how $J_X$ feeds into the heating and ionisation rates.
- Fixed filename typo in `mkdocs.yaml`: `the-dark_ages.md` → `the-dark-ages.md`.
