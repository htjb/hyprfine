# Lyman-$\alpha$ Flux and the Wouthuysen-Field Effect

The Wouthuysen-Field (WF) effect couples the spin temperature of neutral
hydrogen to the kinetic temperature of the gas through Lyman-$\alpha$ photons.
A Lyman-$\alpha$ photon absorbed by a hydrogen atom in the hyperfine ground
state drives it via an intermediate excited state to the other hyperfine
sublevel, effectively scrambling the spin state. When enough Lyman-$\alpha$
photons are present, the spin temperature is pulled toward the gas kinetic
temperature $T_k$.

The strength of this coupling is characterised by the dimensionless coefficient
$x_\alpha$ (see `hyprfine.analytic.coupling_coeffs`):

$$x_\alpha = S_\alpha\, J_\alpha(\nu_\alpha) \times 1.811 \times 10^{11}
\frac{1}{1+z} \frac{2.725\,{\rm K}}{T_{\rm CMB,0}}$$

where $J_\alpha(\nu_\alpha)$ is the Lyman-$\alpha$ specific intensity at the
line-centre frequency $\nu_\alpha = 2.466 \times 10^{15}$ Hz and $S_\alpha
\approx 1$ is an order-unity correction for the spectral shape near line centre,
which we set to 1. Computing $x_\alpha$ therefore reduces to computing
$J_\alpha$.

## The Lyman-$\alpha$ background

Lyman-$\alpha$ photons reach an observer at redshift $z$ from two channels:
direct Ly-$\alpha$ emission from galaxies, and higher Lyman-series photons
(Ly-$n$, $n \geq 2$) emitted at earlier times that redshift into the Ly-$\alpha$
resonance. We follow Munoz et al. (2023), eq. 24, and compute the Lyman-$\alpha$
flux by integrating the star formation rate density (SFRD) over comoving radial
shells:

$$J_\alpha(z) = \frac{(1+z)^2}{4\pi} \int dR\, \dot{\rho}_*\!\left(z'(R)\right)
\varepsilon_\alpha^{\rm tot}(\nu', z', z)$$

where $R$ is comoving distance, $z'(R)$ is the source redshift at distance $R$,
$\nu' = \nu(1+z')/(1+z)$ is the emitted frequency corresponding to the
observed frequency $\nu$, and $\varepsilon_\alpha^{\rm tot}$ is the effective
emissivity accounting for all Lyman transitions that can eventually cascade into
Ly-$\alpha$ photons at the observer's location. The $(1+z)^2$ factor converts
the comoving SFRD to the physical frame appropriate for the intensity integral.
The integral is evaluated over $N_{\rm shells} = 50$ radial shells out to
$z_{\rm max} = 35$ and is implemented in
`hyprfine.analytic.wouthuysen_field.J_alpha`.

## Star formation rate density

The SFRD $\dot{\rho}_*$ is computed by integrating the per-halo star formation
rate over the halo mass function (see `hyprfine.analytic.sfrd`):

$$\dot{\rho}_*(z) = \int d\ln M_h\, M_h \frac{dn}{dM_h}(M_h, z)\, \dot{M}_*(M_h, z)$$

### Halo mass function

We use the Sheth-Tormen (1999) halo mass function:

$$\frac{dn}{dM_h} = \frac{\rho_m}{M_h^2} f_{\rm ST}(\nu)
\left|\frac{d\ln\sigma}{d\ln M_h}\right|$$

where $\nu = \sqrt{q_{\rm ST}}\,\delta_c / \sigma(M_h, z)$, $\delta_c = 1.686$
is the linear overdensity threshold for collapse, $\sigma(M_h, z)$ is the
rms matter fluctuation in a sphere enclosing mass $M_h$ at redshift $z$
(computed via the matter power spectrum emulator in `hyprfine.utils.cosmology`),
and

$$f_{\rm ST}(\nu) = A_{\rm ST}\, \nu \left[1 + \nu^{-2P_{\rm ST}}\right]
e^{-\nu^2/2}$$

with $A_{\rm ST} = 0.3222\sqrt{2/\pi}$, $q_{\rm ST} = 0.85$, and
$P_{\rm ST} = 0.3$.

### Star formation efficiency

The star formation rate per halo is

$$\dot{M}_*(M_h, z) = f_*\!(M_h, z)\, f_b\, \dot{M}_h(M_h, z)$$

where $f_b = \Omega_b / \Omega_m$ is the cosmic baryon fraction and $\dot{M}_h$
is the halo mass accretion rate (see below). The star formation efficiency
$f_*$ follows a double power law with a low-mass suppression:

$$f_*\!(M_h, z) = \frac{2\varepsilon\, f_{\rm duty}(M_h, z)}{(M_h/M_{\rm pivot})^{-\alpha_*}
+ (M_h/M_{\rm pivot})^{-\beta_*}}$$

where $\varepsilon$ sets the overall normalisation, $\alpha_*$ and $\beta_*$
control the slopes above and below the pivot mass $M_{\rm pivot}$, and

$$f_{\rm duty}(M_h, z) = \exp\!\left(-M_{\rm turn}(z)/M_h\right)$$

suppresses star formation in halos below the turnover mass

$$M_{\rm turn}(z) = 3.3 \times 10^7\, M_\odot \left(\frac{1+z}{21}\right)^{-1.5}$$

which approximates the atomic cooling threshold. The free astrophysical
parameters $(\varepsilon, \alpha_*, \beta_*, M_{\rm pivot})$ are collected in
the `astrophysics` namedtuple.

### Halo mass accretion rate

We use the approximation from Correa et al. (2015), which provides better
accuracy over a wider range of cosmologies than earlier Fakhouri et al. (2010)
fitting formula:

$$\dot{M}_h = 71.6\, \frac{M_h}{10^{12}\,M_\odot}\, \frac{h}{0.7}\, f(M_h)
\left[(1+z) - a\right] E(z)$$

where $f(M_h) \propto 1/\sqrt{\sigma^2(M_h/q) - \sigma^2(M_h)}$ encodes the
dependence on the shape of the matter power spectrum, $E(z) = H(z)/H_0$, and
the parameters $z_f$, $q$, and $a$ are determined from the halo mass following
appendix B of Correa et al. (2015).

## Stellar SED and effective emissivity

The intrinsic stellar emissivity $\varepsilon_\alpha(\nu)$ — the number of
Lyman-band photons emitted per unit frequency per unit star formation rate —
is modelled as a double power law (Munoz et al. 2023, eq. 26):

$$\varepsilon_\alpha(\nu) \propto \begin{cases}
0 & \nu < \nu_\alpha \\
(\nu/\nu_\beta)^{\alpha_{\rm low}} & \nu_\alpha \leq \nu < \nu_\beta \\
(\nu/\nu_\beta)^{\alpha_{\rm high}} & \nu_\beta \leq \nu < \nu_{\rm LL}
\end{cases}$$

where $\nu_\alpha = 2.466\times10^{15}$ Hz, $\nu_\beta = 2.922\times10^{15}$ Hz,
and $\nu_{\rm LL} = 3.289\times10^{15}$ Hz are the Lyman-$\alpha$, Lyman-$\beta$
and Lyman-limit frequencies. The default spectral indices are
$\alpha_{\rm low} = 0.14$ (rising toward Ly-$\beta$) and
$\alpha_{\rm high} = -8.0$ (steep drop above Ly-$\beta$). The two segments are
normalised so that 68% of the photon budget falls in the Ly-$\alpha$–Ly-$\beta$
band and 32% above Ly-$\beta$. The overall normalisation $N_\alpha$ is the
total number of Lyman photons emitted per stellar baryon, with a default value
of 9690. This is implemented in
`hyprfine.analytic.wouthuysen_field.calculate_epsilon_alpha_intrinsic`.

### Photon recycling

Higher Lyman series photons ($n \geq 3$) do not automatically produce a
Ly-$\alpha$ photon: they can instead cascade via the $2s$ two-photon channel
and escape without producing a Ly-$\alpha$ resonance photon. The probability
that a photon emitted at level $n$ eventually produces a Ly-$\alpha$ photon is
the recycling fraction $f_{\rm rec}(n)$. We use the values tabulated by
Pritchard & Furlanetto (2006), which are also used in Zeus21 and 21cmFAST.
For $n=2$ (direct Ly-$\alpha$) $f_{\rm rec}=1$; for $n=3$ (Ly-$\beta$)
$f_{\rm rec}=0$; for $n\geq 4$ the values asymptote toward $\approx 0.36$.

The effective emissivity summed over all contributing Lyman lines is
(Munoz et al. 2023, eq. 25):

$$\varepsilon_\alpha^{\rm tot}(z', z) = \varepsilon_\alpha\!\left(\nu
\frac{1+z'}{1+z}\right) \sum_{n=2}^{n_{\rm max}} f_{\rm rec}(n)\, w_\alpha^n(z')$$

where the window function $w_\alpha^n(z') = 1$ when $z' < z_{\rm max}^n(z')$
with

$$1 + z_{\rm max}^n(z') = (1 + z') \frac{1 - (1+n)^{-2}}{1 - n^{-2}}$$

As implemented, this condition is satisfied for all $n \geq 2$ and all
positive $z'$, so in practice the recycling fractions $f_{\rm rec}(n)$ alone
determine which Lyman lines contribute — notably $f_{\rm rec}(3) = 0$ ensures
Ly-$\beta$ photons do not double-count. We sum up to $n_{\rm max} = 23$; the
recycling fractions converge well before this limit. This is implemented in
`hyprfine.analytic.wouthuysen_field.calculate_epsilon_alpha_tot`.
