"""Take power spectrum and make a box."""

import jax
import jax.numpy as jnp

from hyprfine.matterpower import matterpowerspec
from hyprfine.parameters import cosmology
from hyprfine.utils.cosmology import growth_factor

def initialbox(cosmo: cosmology, z: jnp.ndarray,
               Lpix: float, Npix: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a box of density fluctuations from the matter power spectrum.

    Args:
        cosmo: Cosmology parameters.
        z: Redshift.
        Lpix: Pixel size in Mpc/h.
        Npix: Number of pixels in each dimension. 

    Returns:
        Density fluctuations at redshift z.
    """
    # for a box it should really be baryon power spectrum...
    # I think can do this with P_b(k, z) = A_s * k**n_s * T_b(k, z)**2
    # where T_b(k, z) is the baryon transfer function from CAMB/CLASS
    # cosmopower_jax doesn't have this so might need to emulate with astroemu
    k, Pk = matterpowerspec(cosmo, z=0)
    factor = growth_factor(z, cosmo)
    Pk = Pk * factor**2 

    kx = 2 * jnp.pi * jnp.fft.fftfreq(Npix, d=Lpix)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Npix, d=Lpix)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Npix, d=Lpix)

    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing="ij")

    k_mag = jnp.sqrt(kx**2 + ky**2 + kz**2)

    pk_interp = jax.vmap(lambda k_val: jnp.interp(k_val, k, Pk))(k_mag.flatten())

    pk_grid = pk_interp.reshape((Npix, Npix, Npix)) * (1/Lpix) ** 3
    
    white_noise = jax.random.normal(jax.random.PRNGKey(0), (Npix, Npix, Npix))
    noise_k = jnp.fft.fftn(white_noise)
    delta_k = noise_k * jnp.sqrt(pk_grid)
    delta = jnp.fft.ifftn(delta_k).real

    return delta, k_mag, pk_grid.reshape((Npix, Npix, Npix)), k, Pk