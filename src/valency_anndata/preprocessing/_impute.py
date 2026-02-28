import numpy as np
from anndata import AnnData
from typing import Literal, Optional
from sklearn.impute import SimpleImputer, KNNImputer

def impute(
    adata: AnnData,
    *,
    strategy: Literal["zero", "mean", "median", "knn"] = "mean",
    source_layer: Optional[str] = None,
    target_layer: Optional[str] = None,
    overwrite: bool = False,
    params: Optional[dict] = None,
) -> None:
    """
    Impute NaN values in an AnnData matrix and store the result in a layer.

    Uses :class:`sklearn.impute.SimpleImputer` for basic strategies and
    :class:`sklearn.impute.KNNImputer` for ``strategy="knn"``. Any keyword
    arguments accepted by those classes can be passed via ``params``.

    Parameters
    ----------
    adata
        AnnData object.
    strategy
        Imputation strategy:

        - ``"zero"`` — replace NaNs with 0 (``SimpleImputer(strategy="constant", fill_value=0)``)
        - ``"mean"`` — column-wise mean (``SimpleImputer(strategy="mean")``)
        - ``"median"`` — column-wise median (``SimpleImputer(strategy="median")``)
        - ``"knn"`` — k-nearest neighbors (``KNNImputer()``)
    source_layer
        Layer to read from. If None, uses adata.X.
    target_layer
        Layer to write to. Defaults to ``"X_imputed_<strategy>"``.
    overwrite
        Whether to overwrite an existing target layer.
    params
        Extra keyword arguments forwarded directly to the underlying sklearn
        imputer constructor. Common options:

        - ``strategy="knn"``: ``n_neighbors`` (default ``5``),
          ``weights`` (``"uniform"`` or ``"distance"``) —
          see :class:`sklearn.impute.KNNImputer` for the full list.
        - ``strategy="mean"/"median"/"zero"``: ``keep_empty_features``, etc. —
          see :class:`sklearn.impute.SimpleImputer` for the full list.
    """
    if target_layer is None:
        target_layer = f"X_imputed_{strategy}"

    if not overwrite and target_layer in adata.layers:
        return

    # Select source matrix
    if source_layer is None:
        X = adata.X
    else:
        X = adata.layers[source_layer]

    if X is None:
        raise ValueError("No source matrix available for imputation.")

    X = np.asarray(X, dtype=float)
    extra = params or {}

    if strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True, **extra)
    elif strategy in {"mean", "median"}:
        imputer = SimpleImputer(strategy=strategy, keep_empty_features=True, **extra)
    elif strategy == "knn":
        imputer = KNNImputer(**extra)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy!r}")

    adata.layers[target_layer] = imputer.fit_transform(X)
