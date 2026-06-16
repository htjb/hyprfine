import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hyprfine.parameters import cosmology
from hyprfine.box.initialbox import initialbox

def power_spectrum(delta, Lpix, Npix):
    """Spherically averaged power spectrum from a density field box.

    Args:
        delta: Real-space density field, shape (Npix, Npix, Npix).
        Lpix: Pixel size in Mpc/h.
        Npix: Number of pixels per dimension.

    Returns:
        k_bins: Bin centres in h/Mpc.
        Pk: Power spectrum in (Mpc/h)^3.
    """
    L = Lpix * Npix
    V = L ** 3

    delta_k = jnp.fft.fftn(delta)
    # |δ̃|^2 with continuous-FT normalisation: multiply DFT by Lpix^3
    # so P(k) = |δ̃_cont|^2 / V = |DFT|^2 * Lpix^6 / V = |DFT|^2 * Lpix^3 / Npix^3
    power = jnp.abs(delta_k) ** 2 * Lpix ** 3 / Npix ** 3

    kx = 2 * jnp.pi * jnp.fft.fftfreq(Npix, d=Lpix)
    kx, ky, kz = jnp.meshgrid(kx, kx, kx, indexing="ij")
    k_mag = jnp.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    dk = 2 * jnp.pi / L  # fundamental mode
    k_max = jnp.pi / Lpix  # Nyquist
    k_edges = jnp.arange(dk / 2, k_max + dk, dk)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

    Pk = jnp.zeros_like(k_bins)
    for i in range(len(k_bins)):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
        Pk = Pk.at[i].set(jnp.where(mask.sum() > 0, power[mask].mean(), 0.0))

    return k_bins, Pk

cosmo = cosmology()

deltas, k_mag, pk_interp, k, pk = \
    initialbox(cosmo, z=1100.0, Lpix=3.0, Npix=32)

print("Deltas shape:", deltas.shape)
print("K_mag shape:", k_mag.shape)
print("Pk_interp shape:", pk_interp.shape)
print("K shape:", k.shape)
print("Pk shape:", pk.shape)

plt.plot(k, pk)

k, pk = power_spectrum(deltas, Lpix=3.0, Npix=32)
plt.plot(k, pk, label="Box Power Spectrum")
plt.loglog()
#plt.plot(k_mag.flatten(), pk_interp.flatten(), 'o', markersize=2, alpha=0.5)
plt.xlabel("k [h/Mpc]")
plt.ylabel("P(k) [(Mpc/h)^3]")
plt.show()