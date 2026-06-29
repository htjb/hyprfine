In this section of the documentation we outline the modelling choices made in `hyprfine`. The theoretical modelling behind `hyprfine` is not novel and several analytic models for the 21-cm signal exist including [Zeus21](https://github.com/ZeusCosmo/Zeus21) and [ECHO21](https://github.com/shikharmittal04/echo21). Here we detail the specific model choices we made and give a bit of background on how to model the sky-averaged or global 21-cm signal.

Neutral hydrogen consists of an electron and a proton each with their own spin and these spins can be aligned or anti-aligned. When a neutral hydrogen atom transitions from one state to the other it either emits or absorbs a photon with a wavelength of 21-cm. We model the relative number of atoms in each spin state $n_j$ with a statistical temperature which we refer to as the spin temperature $T_s$

$$\frac{n_1}{n_0} = \frac{g_1}{g_0} \exp(- \frac{T_*}{T_s})$$

where $\frac{g_1}{g_0}$ is approximately equal to $3$ and $T_*$ is the excitation energy between the two states.

The relative number of atoms in each state is driven by different radiative fields at different cosmic times and tracing the evolution in $T_s$ can help us understand the expansion of the Universe, star formation and much more. For example at very early cosmic times the spin temperature is mediated entirely by interactions with photons from the CMB, which in turn is coupled to the gas temperature, and we can say that

$$T_s = T_{\rm CMB} = T_k$$

However, as the universe evolves these three temperatures start to deviate from each other with the gas temperature cooling quicker than the radio background and eventually heating up above the radio background due to energy input from various sources. The spin temperature couples to $T_k$ and $T_{\rm CMB}$ with various strengths over time due to different process and that coupling is captured by several coupling coefficients $x_\gamma$, $x_c$ and $x_\alpha$. 

$x_\gamma$ is the strength of the coupling to the radio background and is assumed to be 1 in `hyprfine`, although this can vary in models with excess radio backgrounds. $x_c$ models the coupling between the gas temperature and spin temperature from collisions between different neutral hydrogen atoms. $x_\alpha$ captures the coupling between the gas temperature and spin temperature from the Wouthuysen-Field (WF) effect. The dominant mechanism driving the spin temperature, collisions causing excitations or the WF effect changes with time and the spin temperature as a function of redshift is given by

$$T_s^{-1} = \frac{x_\gamma T_{\rm CMB}^{-1} + x_c T_k^{-1} + x_{\alpha} T_k^{-1}}{x_\gamma + x_c + x_\alpha}$$

where the CMB temperature is given by $T_{\rm CMB}(z) = 2.725 (1 + z)$ and the gas temperature is driven by different processes at different times. This is coded up in `hyprfine.analytic.temperatures`.


We model the 21-cm brightness temperature $T_{21}$ relative to the radio background $T_{\rm CMB}$ according to

$$T_{21} = 54 (1 - x_e) \frac{(1 - Y_{\rm He})}{0.76} \frac{\Omega_b h^2}{0.02242} \sqrt{\frac{0.1424}{\Omega_m h^2}\frac{1+z}{40}}\bigg(1 - \frac{T_{\rm CMB}}{T_s}\bigg)$$

from Mondal et al. (2310.15530) (see `hyprfine.analytic.signal`). $x_e$ is the neutral fraction, $Y_{He}$ is the helium fraction, $\Omega_b$ and $\Omega_m$ are the baryon and matter density parameters and $h$ is $H_0$ Hubble's constant divided by 100 km/s/Mpc.

To analytically model the 21-cm signal during the Dark Ages, Cosmic Dawn and Epoch of reionization we need to estimate $x_c$, $x_\alpha$, $T_k$ and $x_e$.

In this documentation we discuss estimating the 21-cm signal during [the Dark Ages](the-dark-ages.md), [estimating $T_k$ and $x_e$](recombination/hyrec_emulators.md) before the formation of the first stars and galaxies, [modelling the Lyman-$\alpha$ emission](wouthuysen.md) from early galaxies, [the X-ray emission](xrays.md) from early galaxies and [$T_k$ and $x_e$](recombination/cosmic-dawn.md) during the Cosmic Dawn and EoR.