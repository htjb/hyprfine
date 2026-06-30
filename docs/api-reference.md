# API Reference

## Signal generation

The primary entry point for computing the 21-cm signal.

::: hyprfine.analytic.main.generate_signal

---

## Parameters

::: hyprfine.parameters.cosmology

---

## Recombination emulator

::: hyprfine.recombination.emulator.call_hyrec_emulator

---

## Lower-level components

These functions are called internally by `generate_signal` but are exposed
for users who want to compute individual quantities.

### Temperatures

---

::: hyprfine.analytic.temperatures.Tcmb

::: hyprfine.analytic.temperatures.Ts

### Coupling

---

::: hyprfine.analytic.coupling_coeffs.xc

::: hyprfine.analytic.coupling_coeffs.kappa

### 21-cm brightness temperature

---

::: hyprfine.analytic.signal.T21

### Cosmological utilities

--- 

::: hyprfine.utils.cosmology.n_H_tot
