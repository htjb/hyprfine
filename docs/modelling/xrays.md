# X-ray Heating

X-rays from early galaxies — predominantly high-mass X-ray binaries (HMXBs) —
travel far through the largely neutral IGM and deposit energy as heat and
secondary ionisations. Unlike UV photons, which are absorbed within or
immediately around the host galaxy, soft X-rays (0.5–2 keV) can traverse
cosmological distances before being absorbed, making them the dominant heating
mechanism of the neutral IGM during the Cosmic Dawn.

## X-ray background intensity

The X-ray specific intensity $J_X(\nu, z)$ at observed frequency $\nu$ and
redshift $z$ is computed by integrating the attenuated X-ray emissivity of
galaxies at earlier redshifts over comoving radial shells
(see `hyprfine.analytic.xrays.J_X`):

$$J_X(\nu, z) = \frac{(1+z)^2}{4\pi} \int dR\, \dot{\rho}_*\!\left(z'(R)\right)
\varepsilon_X^{\rm tot}(\nu, z', z)$$

where $R$ is comoving distance, $z'(R)$ is the source redshift at distance $R$,
and the SFRD $\dot{\rho}_*$ is the same Sheth-Tormen + Correa et al. model
described in [Lyman-$\alpha$ Flux](wouthuysen.md). The $(1+z)^2$ factor
converts the comoving SFRD to the physical frame. The integral runs over
$N_{\rm shells} = 30$ radial shells out to $z_{\rm max} = 35$.

## Intrinsic X-ray emissivity

The intrinsic X-ray emissivity $\varepsilon_X(\nu)$ — the X-ray power radiated
per unit frequency per unit star formation rate — is modelled as a power law
within the 0.5–2 keV band
(see `hyprfine.analytic.xrays.calculate_epsilon_x_intrinsic`):

$$\varepsilon_X(\nu) \propto \left(\frac{\nu_{\rm keV}}{1\,{\rm keV}}\right)^{\alpha_X}
\qquad \nu_0 \leq \nu_{\rm keV} \leq 2\,{\rm keV}$$

where $\nu_{\rm keV}$ is the photon frequency expressed in keV,
$\alpha_X$ is the spectral index (negative for the falling spectra typical of
HMXBs), and $\nu_0 = 0.5$ keV is the lower band edge. The spectral shape $(\nu_{\rm keV}/1\,{\rm keV})^{\alpha_X}$ is normalised to
integrate to unity over the band in keV, and the result is divided by the
frequency in Hz to give a per-Hz emissivity. The overall scale is set by
$L_{40}$, the X-ray luminosity per unit SFR in units of
$10^{40}$ erg s$^{-1}$ (M$_\odot$ yr$^{-1}$)$^{-1}$. The parameters
$L_{40}$, $\alpha_X$, and $\nu_0$ are free astrophysical parameters collected
in the `astrophysics` namedtuple.

## IGM attenuation

X-rays are attenuated by photoionisation of neutral hydrogen along the line of
sight. The optical depth between observer redshift $z$ and source redshift $z'$
is (see `hyprfine.analytic.xrays.tau_X`):

$$\tau_X(\nu, z, z') = \int_z^{z'} \frac{c\, n_H(z'')\, \sigma_X(\nu'')}{H(z'')(1+z'')}\, dz''$$

where $\nu'' = \nu(1+z'')/(1+z)$ is the photon frequency at intermediate
redshift $z''$ and the HI photoionisation cross section is approximated as

$$\sigma_X(\nu) = \sigma_0 \left(\frac{\nu}{\nu_{\rm HI}}\right)^{-3}
\qquad \nu \geq \nu_{\rm HI}$$

with $\sigma_0 = 6.3 \times 10^{-18}$ cm$^2$ and
$\nu_{\rm HI} = 3.288 \times 10^{15}$ Hz (the 13.6 eV ionisation threshold).
Harder X-rays have smaller cross sections and therefore longer mean free paths,
so higher-energy photons contribute disproportionately to IGM heating at
large distances. The attenuated emissivity is then

$$\varepsilon_X^{\rm tot}(\nu, z', z) = \varepsilon_X\!\left(\nu
\frac{1+z'}{1+z}\right) e^{-\tau_X(\nu, z, z')}$$

## Heating and ionisation

The X-ray specific intensity $J_X$ feeds directly into the IGM temperature and
ionisation ODEs described in
[The IGM During the Cosmic Dawn](recombination/cosmic-dawn.md). The volumetric
X-ray heating rate is

$$Q_X(z) = 4\pi \int d\nu\, J_X(\nu,z)\, n_H \sigma_X(\nu)\, f_{\rm heat}(x_e)
\left(1 - \frac{h\nu_{\rm HI}}{h\nu}\right)$$

and the X-ray secondary ionisation rate per H atom is

$$\Gamma_X(z) = 4\pi\, f_{\rm ion}(x_e) \int d\nu\,
\frac{J_X(\nu,z)\, \sigma_X(\nu)}{h\nu}$$

where $f_{\rm heat}(x_e)$ and $f_{\rm ion}(x_e)$ are the Shull & van Steenberg
(1985) energy deposition fractions described in
[The IGM During the Cosmic Dawn](recombination/cosmic-dawn.md). To avoid
re-evaluating $J_X$ at every ODE step, it is precomputed on a redshift grid
and interpolated within the integrator.
