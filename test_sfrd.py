import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.sfrd import dn_dmh, mean_sfrd, sfrd, fstar
from hyprfine.utils.cosmology import growth_factor, rhom, sigma, sigma0
from hyprfine.parameters import cosmology
from hyprfine.matterpower import matterpowerspec

cosmo = cosmology(
    H0=67.36,
    Omega_m=0.315,
    Omega_b=0.049,
    Omega_c=0.266,
    Y_He=0.245,
    ns=0.97,
    ln1010As=3.044,
)  # Example cosmology parameters

epsilon = 0.1
alpha_star = 0.5
beta_star = -0.5
M_pivot = 3e11


Mh = 10 ** jnp.linspace(8, 13, 100)

z_values = jnp.arange(10, 31, 1)

for i in z_values:
    plt.plot(Mh, dn_dmh(Mh, cosmo, z=i))
plt.loglog()
plt.xlabel("Halo Mass $M_h$ ($M_\\odot$)")
plt.ylabel("Halo Mass Function $dn/dM_h$ (Mpc$^{-3}$ $M_\\odot^{-1}$)")
plt.grid()
plt.savefig("halo_mass_function.png", dpi=300)
plt.show()
plt.close()

mean_sfrd_values = jnp.array(
    [
        mean_sfrd(
            cosmo=cosmo,
            z=i,
            epsilon=epsilon,
            alpha_star=alpha_star,
            beta_star=beta_star,
            M_pivot=M_pivot,
            Mh=Mh,
        )
        for i in z_values
    ]
)

plt.plot(z_values, 10**3 * mean_sfrd_values, label="Mean SFRD")
plt.yscale("log")
plt.xlabel("Redshift $z$")
plt.ylabel("Star Formation Rate Density (M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$)")
plt.title("Star Formation Rate Density vs Redshift")
plt.grid()
plt.savefig("mean_sfrd.png", dpi=300)
plt.show()
plt.close()


R = jnp.linspace(0.1, 10, 100)

sfrd_values = [
    sfrd(
        cosmo=cosmo,
        z=i,
        epsilon=epsilon,
        alpha_star=alpha_star,
        beta_star=beta_star,
        M_pivot=M_pivot,
        Mmin=1e8,
        Mmax=1e13,
        R=R,
        key=jax.random.PRNGKey(0),
    )
    for i in range(10)
]

plt.plot(R, sfrd_values[0], label="SFRD at z=0")
plt.plot(R, sfrd_values[5], label="SFRD at z=5")
plt.plot(R, sfrd_values[9], label="SFRD at z=9")
plt.xlabel("Smoothing Scale $R$ (Mpc)")
plt.ylabel("Star Formation Rate Density (M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$)")
plt.title("Star Formation Rate Density vs Smoothing Scale")
plt.legend()
plt.grid()
plt.show()
