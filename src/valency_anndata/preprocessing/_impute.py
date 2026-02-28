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
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "uniform",
) -> None:
    """
    Impute NaN values in an AnnData matrix and store the result in a layer.

    Uses :class:`sklearn.impute.SimpleImputer` for basic strategies and
    :class:`sklearn.impute.KNNImputer` for k-nearest neighbors imputation.

    Parameters
    ----------
    adata
        AnnData object.
    strategy
        Imputation strategy:
        - "zero": replace NaNs with 0
        - "mean": column-wise mean
        - "median": column-wise median
        - "knn": k-nearest neighbors imputation
    source_layer
        Layer to read from. If None, uses adata.X.
    target_layer
        Layer to write to. Defaults to "X_imputed_<strategy>".
    overwrite
        Whether to overwrite an existing target layer.
    n_neighbors
        Number of neighbors to use for KNN imputation. Only used when
        ``strategy="knn"``.
    weights
        Weight function for KNN imputation. ``"uniform"`` weights all
        neighbors equally; ``"distance"`` weights by inverse distance.
        Only used when ``strategy="knn"``.
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

    if strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    elif strategy in {"mean", "median"}:
        imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy!r}")

    adata.layers[target_layer] = imputer.fit_transform(X)
