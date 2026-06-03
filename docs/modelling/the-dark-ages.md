The Dark Ages cover a period between the formation of the CMB and around $z=30$ when the first stars and galaxies formed. The 21-cm signal during this period has no dependence on $x_\alpha$, the coupling coefficient between the spin temperature and the Lyman-$\alpha$ background, and 


$$T_s^{-1} = \frac{T_{\rm CMB}^{-1} + x_c T_k^{-1}}{1 + x_c}$$.

$x_c$ tells us the strength of coupling between the gas, which is cooling adiabatically during this period, and the spin temperature from collisions between neutral hydrogen atoms, neutral hydrogen atoms and free electrons and neutral hydrogen and free protons. These collisions transfer energy between the neutral hydrogen atoms which excites the transition from one spin state to the other. Because the gas is cooling at a quicker rate than the CMB the 21-cm temperature appears in absorption during this period. The rate of collisions is driven largely by the expansion rate of the Universe and the density of matter.

The coupling coefficient (see `hyprfine.analytic.coupling_coeffs`) is given by 

$$x_c = \frac{T_* \kappa(T_k, x_e) n_H}{A_{10} T_{\rm CMB}(z)}$$

where $n_H$ is the mean hydrogen number density (see `hyprfine.utils.cosmology`) given by

$$n_H = \frac{(1 - Y_{He}) \bar{\rho}}/ m_p$$

in m$^{-3}$ where $m_p$ is the proton mass and 

$$\bar{\rho} = \frac{3 H_0^3}{8\pi G} \Omega_b (1+z)**3$$

is the mean baryon density of the Universe at redshift $z$.

$\kappa$ is the rate coefficient for coupling and can be broken down into rate coefficients for H-H collisions, H-e collisions and H-p collisions. Quantum mechanical calculations are needed to calculate these rate coefficents but the results are tabulated in the literature and fitted to as a function of the gas temperatrue. We take these from [Moazzenzadeh and  Firouzjaee 2021](https://arxiv.org/abs/2108.00115) who gives

$$\kappa_{HH} = 3.1\times 10^{-11} T_k^{0.357} \exp(-\frac{32}{T_k}) 10^{-6}$$

$$\log_{10} \kappa_{He} = -9.607 + 0.5 \log_{10} T_k \exp(-\frac{\log_{10}(T_k)**4}{1800}) -6$$

in m$^{3}$/s. $\kappa$ essentially tells us how efficient the collisions are in producing a transition from one spin state to the other. In `hyprfine` $\kappa_{Hp}$ is set to be equal to $\kappa_{He}$. In practice the two rates differ by a factor related to the ratio of the proton and electron masses but they are subdominant effects (see the top right hand figure below) compared to $\kappa_{HH}$ and so it is sufficient to equate the two.

With $\kappa$ we can calculate $x_c$ which is a function of $H_0$, $\Omega_b$, $Y_{He}$ and $\Omega_c$ through $T_k$ and $x_e$. $x_c$ is shown in the top left hand panel in the figure below. The bottom left hand panel shows how $T_s$ is coupled to $T_k$ at high redshifts but then drifts back to $T_{\rm CMB}$ as the universe expands and collisional coupling becomes inefficient ($x_c$ tends to 0). The resulting 21-cm signal is shown in the bottom right panel.

In order to calculate $x_c$ and $T_{21}$, we need to know how the gas temperature $T_k$ and the free electron fraction $x_e$ evolve over time. We discuss how these are calculated [here](docs/modelling/recombination/hyrec_emulators.md).

![Dark Ages Signal](figures/dark-ages.png)

