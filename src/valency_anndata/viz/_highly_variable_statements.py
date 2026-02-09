import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Import Scanpy internals
from scanpy.plotting._utils import savefig_or_show
from scanpy._settings import settings


def highly_variable_statements(
    adata: AnnData,
    *,
    log: bool = False,
    show: bool | None = None,
    save: str | None = None,
) -> None:
    """
    Plot normalized and raw dispersions for statements identified as highly variable.

    Analogous to [scanpy.pl.highly_variable_genes][] for single-cell data. Creates a
    two-panel scatter plot showing normalized dispersion (left) and raw dispersion (right)
    against the binning variable used in `val.preprocessing.highly_variable_statements`.
    Highly variable statements are highlighted in black, others in grey.

    Parameters
    ----------
    adata
        AnnData object that has been processed with
        `val.preprocessing.highly_variable_statements`.
    log
        If True, use log scale for both axes. Default is False.
    show
        If True, display the plot. If None, defaults to Scanpy's `settings.autoshow`.
    save
        File path to save the figure. If provided, figure is saved instead of shown.

    Examples
    --------
    ```py
    import valency_anndata as val
    adata = val.datasets.aufstehen()
    val.preprocessing.highly_variable_statements(adata, n_top_statements=50)
    val.viz.highly_variable_statements(adata)
    ```

    Use log scale for better visibility:

    ```py
    val.viz.highly_variable_statements(adata, log=True)
    ```
    """

    if "highly_variable" not in adata.uns:
        raise ValueError(
            "No highly variable statement metadata found. "
            "Run `val.preprocessing.highly_variable_statements` first."
        )

    result = adata.var
    hv_meta = adata.uns["highly_variable"]

    # Which statements are marked highly variable
    statement_subset = result["highly_variable"].values

    # Means for x-axis (use the same column as `bin_by`)
    means = result[hv_meta.get("bin_by", "coverage")].values

    # dispersions (raw) and dispersions_norm (z-score within bins)
    dispersions = result["dispersions"].values
    dispersions_norm = result["dispersions_norm"].values

    # Setup figure
    size = rcParams["figure.figsize"]
    plt.figure(figsize=(2 * size[0], size[1]))
    plt.subplots_adjust(wspace=0.3)

    for idx, d in enumerate([dispersions_norm, dispersions]):
        plt.subplot(1, 2, idx + 1)
        for label, color, mask in zip(
            ["highly variable statements", "other statements"],
            ["black", "grey"],
            [statement_subset, ~statement_subset],
        ):
            x = means[mask]
            y = d[mask]
            plt.scatter(x, y, label=label, c=color, s=5)

        if log:
            plt.xscale("log")
            plt.yscale("log")
            y_min = np.nanmin(d)
            y_min = 0.95 * y_min if y_min > 0 else 1e-1
            plt.xlim(0.95 * np.nanmin(means), 1.05 * np.nanmax(means))
            plt.ylim(y_min, 1.05 * np.nanmax(d))

        if idx == 0:
            plt.legend()
        plt.xlabel(f"{hv_meta.get('bin_by', 'coverage')}")
        plt.ylabel(f"{'normalized ' if idx == 0 else ''}dispersion")

    # determine whether to show (default Scanpy behavior)
    show = settings.autoshow if show is None else show
    savefig_or_show("highly_variable_statements", show=show, save=save)
