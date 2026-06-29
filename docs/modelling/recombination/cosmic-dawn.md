# The IGM During the Cosmic Dawn

During the Cosmic Dawn and Epoch of Reionisation (EoR), roughly $z \lesssim 35$,
the first stars and galaxies begin to form. Their radiation drives significant
changes in the intergalactic medium (IGM): X-rays heat the gas, UV photons
ionise it, and Lyman-$\alpha$ photons couple the spin temperature to the
kinetic temperature of the gas. We therefore need to track how $T_k$ and $x_e$
evolve under these influences.

## Initial conditions

At $z = 35$ we take $T_k$ and $x_e$ from the
[HYREC-2 emulator](hyrec_emulators.md), which accurately captures the
recombination history down to this redshift before any significant star
formation has occurred. These values serve as initial conditions for the ODE
system described below.

## Coupled ODEs for $T_k$, $x_{e,\rm bg}$, and $Q$

From $z = 35$ down to the end of reionisation, we integrate three coupled
first-order ordinary differential equations. The integration is performed using
the [diffrax](https://docs.kidger.site/diffrax/) library with the implicit
Kvaerno5 solver. See `hyprfine.recombination.odes`.

The mean free electron fraction is split into two physically distinct
contributions:

- **$x_{e,\rm bg}$** — the "background" ionisation fraction of the diffuse IGM
  outside HII bubbles. This includes the cosmological recombination residual
  handed off from HYREC-2 and the boost from X-ray secondary ionisations. It
  evolves under quadratic (homogeneous) recombination.
- **$Q$** — the volume-filling factor of fully-ionised HII bubbles driven by UV
  photons from early galaxies. Following Madau et al. (1999), recombinations
  inside bubbles scale linearly with $Q$ (bubble model).

The total mean ionisation fraction entering the 21-cm brightness temperature is

$$x_e = x_{e,\rm bg} + Q$$

### Temperature evolution

$$\frac{dT_k}{dz} = \frac{2T_k}{1+z} + \frac{dt}{dz}\left(\dot{T}_{\rm Compton} + \dot{T}_{X}\right)$$

where $dt/dz = -1/[H(z)(1+z)]$. The three terms represent:

**Adiabatic cooling.** The $2T_k/(1+z)$ term captures the adiabatic cooling of
the gas as the Universe expands. In the absence of any heating the gas
temperature falls as $T_k \propto (1+z)^2$.

**Compton heating.** Residual free electrons scatter off CMB photons,
transferring energy between the radiation field and the gas:

$$\dot{T}_{\rm Compton} = \frac{8 \sigma_T u_{\rm CMB} x_e}{3 m_e c (1 + x_e + Y_{\rm He}/4)} \left(T_{\rm CMB} - T_k\right)$$

where $u_{\rm CMB} = (4\sigma_{\rm SB}/c)T_{\rm CMB}^4$ is the CMB energy
density and $\sigma_T$ is the Thomson cross section. This term drives $T_k$
toward $T_{\rm CMB}$ at early times when $x_e$ is still significant.

**X-ray heating.** X-ray photons from the first galaxies are partially absorbed
by the neutral IGM. The fraction of absorbed energy that goes into heat is
$f_{\rm heat}(x_e)$ (see below), and the volumetric heating rate is

$$Q_X = 4\pi \int d\nu\, J_X(\nu, z)\, n_H \sigma_X(\nu) f_{\rm heat}(x_e)
\left(1 - \frac{h\nu_{\rm HI}}{h\nu}\right)$$

where $J_X$ is the X-ray specific intensity (discussed in
[X-ray Heating](../xrays.md)), $\sigma_X(\nu) \propto \nu^{-3}$ is the HI
photoionisation cross section, and $\nu_{\rm HI}$ is the HI ionisation
threshold frequency. The corresponding contribution to the temperature
evolution is

$$\dot{T}_X = \frac{Q_X}{1.5\, k_B (1 + x_e + Y_{\rm He}/4)\, n_H}$$

### Background ionisation ($x_{e,\rm bg}$)

The background IGM is ionised by X-ray secondaries and recombines in a
homogeneous (quadratic) fashion:

$$\frac{dx_{e,\rm bg}}{dz} = \frac{dt}{dz} \left(
  -\alpha_B\, n_H\, x_{e,\rm bg}^2
  + \Gamma_X (1 - x_e)
\right)$$

Here $\alpha_B = 2.6 \times 10^{-13} (T_k / 10^4\,{\rm K})^{-0.76}$ cm$^3$
s$^{-1}$ is the case-B recombination coefficient and $x_e = x_{e,\rm bg} + Q$
is the total ionisation fraction. The quadratic $x_{e,\rm bg}^2$ term arises
from $n_e n_p \propto x_{e,\rm bg}^2$ in the diffuse neutral IGM. The X-ray
secondary ionisation rate is

$$\Gamma_X = 4\pi\, f_{\rm ion}(x_{e,\rm bg}) \int d\nu\,
\frac{J_X(\nu,z)\, \sigma_X(\nu)}{h\nu}$$

### HII bubble filling factor ($Q$)

UV (Lyman continuum) photons from early galaxies create fully-ionised HII
bubbles. Following Madau et al. (1999), the volume-filling factor $Q$ evolves as

$$\frac{dQ}{dz} = \frac{dt}{dz} \left(
  \frac{\dot{n}_{\rm ion}}{n_H} - C_{\rm HII}\, \alpha_B\, n_H\, Q
\right)$$

The recombination term is linear in $Q$ because recombinations only occur inside
the already-ionised regions (volume fraction $Q$), where $n_e = n_p = n_H$.
The clumping factor $C_{\rm HII} = \max(1,\, 2.9\,[(1+z)/6]^{-1.1})$ accounts
for sub-resolution density inhomogeneities inside bubbles.

The UV ionising photon rate density is (see `hyprfine.analytic.ionization`)

$$\dot{n}_{\rm ion} = \frac{f_{\rm esc}\, N_{\rm ion}\, \dot{\rho}_*(z)}{\bar{m}_b}$$

where $f_{\rm esc}$ is the ionising-photon escape fraction, $N_{\rm ion}$ is the
number of ionising photons per stellar baryon, $\dot{\rho}_*$ is the SFRD (see
[Lyman-$\alpha$ Flux](../wouthuysen.md)), and $\bar{m}_b = 1.22 m_p$.

## Heating and ionisation fractions

The fractions of absorbed X-ray energy that go into heating and secondary
ionisation depend on the ionisation state of the gas. We use the fitting formulae from Shull & van Steenberg (1985), Table 1
(300 eV primary electron), as implemented in 21cmFAST:

$$f_{\rm heat}(x_e) = 0.9971 \left[ 1 - (1 - x_e^{0.2663})^{1.3163} \right]$$

$$f_{\rm ion}(x_e) = 0.3908 \left( 1 - x_e^{0.4092} \right)^{1.7592}$$

In a nearly neutral IGM ($x_e \ll 1$) the absorbed X-ray energy is shared
roughly equally between heating and ionisation, with a small residual fraction
going into collisional excitation. As the IGM ionises, a larger fraction goes
into heating. These fractions are implemented in `hyprfine.recombination.odes`.

## Coupling back to the signal

Once the ODE system is solved, the output $T_k(z)$ and $x_e(z)$ are stitched
onto the HYREC emulator output at $z = 35$. The combined history feeds into the
spin temperature and 21-cm brightness temperature calculation described in
[Global Signal](../intro.md).
