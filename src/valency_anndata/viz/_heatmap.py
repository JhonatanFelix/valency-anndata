from __future__ import annotations

import numpy as np
from anndata import AnnData
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colormaps as _mpl_colormaps

# Register a brighter discrete-friendly variant of RdYlGn.
_RdYlGnBright = ListedColormap(["#d73027", "#ffff00", "#1a9850"], name="RdYlGnBright")
try:
    _mpl_colormaps.register(_RdYlGnBright)
except ValueError:
    pass  # already registered


def heatmap(
    adata: AnnData,
    groupby: str | None = None,
    cmap: str = "RdYlGn",
    discrete: bool = False,
    show: bool = True,
    **kwargs,
):
    """
    Plot a vote-matrix heatmap with Polis-friendly defaults.

    A thin wrapper around :func:`scanpy.pl.heatmap` that adds optional discrete
    colorbar labelling and removes the requirement for a ``groupby`` column.

    Parameters
    ----------
    adata
        AnnData object with participants as observations and statements as variables.
    groupby
        Column in ``adata.obs`` to group participants by. When ``None``,
        participants are shown in their current index order with no grouping.
    cmap
        Colormap name. Defaults to ``"RdYlGn"``. Also accepts ``"RdYlGnBright"``,
        a custom :class:`~matplotlib.colors.ListedColormap` with fully saturated
        red, yellow, and green (``["#d73027", "#ffff00", "#1a9850"]``).
    discrete
        When ``True``, renders a segmented colorbar with labelled ticks
        (``"disagree (-1)"``, ``"pass (0)"``, ``"agree (+1)"``), using a
        :class:`~matplotlib.colors.BoundaryNorm` with boundaries at
        ``[-1.5, -0.5, 0.5, 1.5]``.
    show
        Whether to call ``plt.show()`` at the end. Set to ``False`` to get back
        the axes dict for further customisation.
    **kwargs
        Additional keyword arguments forwarded to :func:`scanpy.pl.heatmap`.

    Returns
    -------
    None or dict
        When ``show=True`` (default) returns ``None``. When ``show=False``,
        returns the axes dictionary from :func:`scanpy.pl.heatmap`.

    Examples
    --------

    ```py
    adata = val.datasets.polis.load("https://pol.is/report/r29kkytnipymd3exbynkd")

    val.viz.heatmap(adata, discrete=True)
    ```
    """
    import scanpy as sc
    import matplotlib.pyplot as plt

    _dummy_col = None
    if groupby is None:
        _dummy_col = "__heatmap_dummy__"
        adata.obs[_dummy_col] = "all"
        adata.obs[_dummy_col] = adata.obs[_dummy_col].astype("category")
        groupby = _dummy_col

    if discrete:
        ticks = np.array([-1.0, 0.0, 1.0])
        boundaries = np.array([-1.5, -0.5, 0.5, 1.5])
        norm = BoundaryNorm(boundaries, ncolors=plt.get_cmap(cmap).N)
        # scanpy forbids passing norm alongside vmin/vmax/vcenter
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)
        kwargs.pop("vcenter", None)
        kwargs["norm"] = norm

    axes_dict = sc.pl.heatmap(
        adata,
        var_names=adata.var_names,
        groupby=groupby,
        cmap=cmap,
        show=False,
        **kwargs,
    )

    if _dummy_col is not None:
        del adata.obs[_dummy_col]
        # Hide the uninformative groupby axis strip
        if axes_dict and "groupby_ax" in axes_dict:
            axes_dict["groupby_ax"].set_visible(False)

    if discrete:
        ticklabels = ["disagree (-1)", "pass (0)", "agree (+1)"]
        # scanpy renders the colorbar via plt.colorbar(image, cax=...), so the
        # Colorbar object is stored on the image artist, not on the QuadMesh.
        cbar = None
        for ax in plt.gcf().axes:
            for img in ax.images:
                cb = getattr(img, "colorbar", None)
                if cb is not None:
                    cbar = cb
                    break
            if cbar is not None:
                break
        if cbar is not None:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)

    if show:
        plt.show()
        return None

    return axes_dict
