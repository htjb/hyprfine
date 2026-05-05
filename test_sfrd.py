import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hyprfine.analytic.sfrd import dn_dmh, mean_sfrd, sfrd, fstar
from hyprfine.parameters import astrophysics, cosmology

cosmo = cosmology(
    H0=67.36,
    Omega_m=0.315,
    Omega_b=0.049,
    Omega_c=0.266,
    Omega_r=9e-5,
    Y_He=0.245,
    ns=0.97,
    ln1010As=3.044,
)  # Example cosmology parameters

astro = astrophysics(
    epsilon=0.1,
    alpha_star=0.5,
    beta_star=-0.5,
    M_pivot=3e11,
    f_esc=0.15,
    N_ion=5000.0,
)

Mh = 10 ** jnp.linspace(8, 13, 1000)

z_values = jnp.arange(10, 31, 1)

from hyprfine.utils.cosmology import sigma
Mh_test = jnp.array([1e8, 1e10, 1e12])
print("sigma at z=10:", sigma(Mh_test, cosmo, 10.0))
print("sigma at z=0:", sigma(Mh_test, cosmo, 0.0))

from hyprfine.utils.cosmology import growth_factor
for i in [10, 15, 20, 25, 30]:
     growth = growth_factor(i, cosmo)
     plt.plot(Mh, dn_dmh(Mh, cosmo, z=i), label=f"z={i}, D(z)={growth:.3f}")
plt.loglog()
plt.legend()
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
            astro=astro,
            Mh=Mh,
        )
        for i in z_values
    ]
)

z = jnp.array([10, 15, 20, 25, 30])
for i in z:
    plt.plot(Mh, fstar(astro, Mh, i), label=f"f_star at z={i}")
plt.xlabel("Halo Mass $M_h$ ($M_\\odot$)")
plt.ylabel("Star Formation Efficiency $f_*$")
plt.title("Star Formation Efficiency vs Halo Mass")
plt.loglog()
plt.legend()
plt.grid()
plt.savefig("fstar.png", dpi=300)
plt.show()
plt.close()

plt.plot(z_values, 10**3 * mean_sfrd_values, label="Mean SFRD")
plt.yscale("log")
plt.xlabel("Redshift $z$")
plt.ylabel("Star Formation Rate Density ($10^3$ M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$)")
plt.title("Star Formation Rate Density vs Redshift")
plt.grid()
#plt.ylim(bottom=1e-4)
plt.savefig("mean_sfrd.png", dpi=300)
plt.show()
plt.close()


# R = jnp.linspace(0.1, 10, 100)

# sfrd_values = [
#     sfrd(
#         cosmo=cosmo,
#         z=i,
#         astro=astro,
#         Mmin=1e8,
#         Mmax=1e13,
#         R=R,
#         key=jax.random.PRNGKey(0),
#     )
#     for i in range(10)
# ]

# plt.plot(R, sfrd_values[0], label="SFRD at z=0")
# plt.plot(R, sfrd_values[5], label="SFRD at z=5")
# plt.plot(R, sfrd_values[9], label="SFRD at z=9")
# plt.xlabel("Smoothing Scale $R$ (Mpc)")
# plt.ylabel("Star Formation Rate Density (M$_\\odot$ yr$^{-1}$ Mpc$^{-3}$)")
# plt.title("Star Formation Rate Density vs Smoothing Scale")
# plt.legend()
# plt.grid()
# plt.show()
