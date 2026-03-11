"""Custom normalisation pipelines for HYREC emulator training."""

import jax
import jax.numpy as jnp
from astroemu.normalisation import NormalisationPipeline


class focus_on(NormalisationPipeline):
    """Redistribute the redshift grid.

    Interpolates spectra from the data's uniform z grid to a new grid
    with higher density between z_focus_lo and z_focus_hi.

    Args:
        z_min: Minimum redshift of the data grid.
        z_max: Maximum redshift of the data grid.
        z_focus_lo: Lower bound of the focus region.
        z_focus_hi: Upper bound of the focus region.
        n_background: Number of uniformly spaced background points
            spanning [z_min, z_max].
        n_focus: Number of extra points concentrated in
            [z_focus_lo, z_focus_hi].
    """

    def __init__(
        self,
        z_min: float = 1.0,
        z_max: float = 8000.0,
        z_focus_lo: float = 700.0,
        z_focus_hi: float = 1500.0,
        n_background: int = 2000,
        n_focus: int = 1500,
    ) -> None:
        """Create the new redshift grid."""
        z_bg = jnp.linspace(z_min, z_max, n_background)
        z_fg = jnp.linspace(z_focus_lo, z_focus_hi, n_focus)
        self._new_grid = jnp.array(jnp.unique(jnp.concatenate([z_bg, z_fg])))

    def forward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Interpolate y from the uniform grid to the focused grid.

        Args:
            y: Spectrum array, shape (batch, len_x).
            x: Redshift array, shape (batch, len_x); all rows identical.
            params: Input parameters, shape (batch, n_params).

        Returns:
            y interpolated onto focused grid, tiled focused x, params.
        """
        x_row = x[0]
        y_new = jax.vmap(lambda yi: jnp.interp(self._new_grid, x_row, yi))(y)
        x_new = jnp.tile(self._new_grid, (x.shape[0], 1))
        return y_new, x_new, params

    def backward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Pass-through — values remain on the focused grid.

        Args:
            y: Spectrum array.
            x: Redshift array.
            params: Parameters array.

        Returns:
            Unchanged y, x, and params.
        """
        return y, x, params


class downsample(NormalisationPipeline):
    """Downsample the grid.

    Args:
        z_min: Minimum redshift of the data grid.
        z_max: Maximum redshift of the data grid.
        n: Number of points in the downsampled grid.
    """

    def __init__(
        self,
        z_min: float = 1.0,
        z_max: float = 8000.0,
        n: int = 2000,
    ) -> None:
        """Create the new redshift grid."""
        self.z_down = jnp.linspace(z_min, z_max, n)

    def forward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Interpolate y from the uniform grid to the focused grid.

        Args:
            y: Spectrum array, shape (batch, len_x).
            x: Redshift array, shape (batch, len_x); all rows identical.
            params: Input parameters, shape (batch, n_params).

        Returns:
            y interpolated onto downsampled grid, tiled downsampled x, params.
        """
        x_row = x[0]
        y_new = jax.vmap(lambda yi: jnp.interp(self.z_down, x_row, yi))(y)
        x_new = jnp.tile(self.z_down, (x.shape[0], 1))
        return y_new, x_new, params

    def backward(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Pass-through — values remain on the focused grid.

        Args:
            y: Spectrum array.
            x: Redshift array.
            params: Parameters array.

        Returns:
            Unchanged y, x, and params.
        """
        return y, x, params
