from anndata import AnnData
import numpy as np
import pandas as pd

def highly_variable_statements(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_bins: int | None = 1,
    min_disp: float | None = None,
    max_disp: float | None = None,
    min_cov: int | None = 2,
    max_cov: int | None = None,
    n_top_statements: int | None = None,
    subset: bool = False,
    inplace: bool = True,
    key_added: str = "highly_variable",
    variance_mode: str = "overall",  # "overall", "valence", "engagement"
    bin_by: str = "coverage",        # "coverage", "p_engaged", "mean_valence", "mean_abs_valence"
):
    """
    Identify highly variable statements in a vote matrix (AnnData).

    Analogous to [scanpy.pp.highly_variable_genes][] for single-cell data, this function
    identifies statements with high variability across participants. The function computes
    various dispersion metrics, normalizes them within bins, and marks statements as highly
    variable based on user-defined criteria.

    Parameters
    ----------
    adata
        AnnData object containing vote matrix.
    layer
        Layer to use for computation. If None, uses `adata.X`.
    n_bins
        Number of bins for dispersion normalization. Values <=1 or None disable binning.
        Default is 1 (no binning).
    min_disp
        Minimum normalized dispersion threshold for selecting highly variable statements.
        Only used if `n_top_statements` is None.
    max_disp
        Maximum normalized dispersion threshold for selecting highly variable statements.
        Only used if `n_top_statements` is None.
    min_cov
        Minimum coverage (number of non-NaN votes) required for a statement.
        Default is 2.
    max_cov
        Maximum coverage threshold for selecting highly variable statements.
        Only used if `n_top_statements` is None.
    n_top_statements
        Select this many top statements by normalized dispersion. If provided, overrides
        `min_disp`, `max_disp`, and `max_cov` filters.
    subset
        If True, subset the AnnData object to highly variable statements.
    inplace
        If True, add results to `adata.var` and `adata.uns[key_added]`.
        If False, return results as DataFrame.
    key_added
        Key under which to store the highly variable boolean mask in `adata.var`
        and metadata in `adata.uns`. Default is "highly_variable".
    variance_mode
        Which variance metric to use for computing dispersion:
        - "overall": variance of raw votes (including NaN as missing)
        - "valence": variance of engaged votes only (excluding passes/NaN)
        - "engagement": variance of engagement (1 if ±1, 0 if pass)
        Default is "overall".
    bin_by
        Variable to bin on for normalization. Options:
        - "coverage": number of non-NaN votes
        - "p_engaged": proportion of engaged votes (±1)
        - "mean_valence": average valence of engaged votes
        - "mean_abs_valence": absolute value of mean valence
        Default is "coverage".

    Returns
    -------
    pd.DataFrame | None
        If `inplace=False`, returns a DataFrame with columns:
        `coverage`, `mean_valence`, `mean_abs_valence`, `p_engaged`,
        `bin_idx`, `var_overall`, `var_valence`, `var_engagement`,
        `dispersions`, `dispersions_norm`, and a boolean column named by `key_added`.
        If `inplace=True`, modifies `adata` in place and returns None.

    Examples
    --------
    Select top 50 most variable statements:

    ```py
    import valency_anndata as val
    adata = val.datasets.aufstehen()
    val.preprocessing.highly_variable_statements(adata, n_top_statements=50)
    ```

    Use normalized dispersion thresholds with binning:

    ```py
    val.preprocessing.highly_variable_statements(
        adata,
        n_bins=10,
        min_disp=0.5,
        min_cov=5,
        bin_by="coverage"
    )
    ```

    Focus on valence variance instead of overall variance:

    ```py
    val.preprocessing.highly_variable_statements(
        adata,
        n_top_statements=100,
        variance_mode="valence"
    )
    ```

    Run multiple times with different settings using `key_added`:

    ```py
    # Identify top 50 statements
    val.preprocessing.highly_variable_statements(
        adata,
        n_top_statements=50,
        key_added="highly_variable_top50"
    )
    # Also identify top 100 statements
    val.preprocessing.highly_variable_statements(
        adata,
        n_top_statements=100,
        key_added="highly_variable_top100"
    )
    # Now you can use either mask with recipe_polis
    val.tools.recipe_polis(adata, mask_var="highly_variable_top50")
    ```
    """

    # ---- 0. select matrix ---------------------------------------------
    X = adata.layers[layer] if layer is not None else adata.X
    X = np.asarray(X)
    n_statements = X.shape[1]

    # ---- 1. coverage and engagement -----------------------------------
    coverage = np.sum(~np.isnan(X), axis=0)
    engaged = (~np.isnan(X)) & (X != 0)
    p_engaged = engaged.sum(axis=0) / np.maximum(coverage, 1)

    # average valence for engaged votes only
    mean_valence = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        vals = X[engaged[:, j], j]
        if vals.size > 0:
            mean_valence[j] = np.mean(vals)

    # optional: absolute version
    mean_abs_valence = np.abs(mean_valence)

    # ---- 2. compute variances -----------------------------------------
    # overall variance
    var_overall = np.nanvar(X, axis=0, ddof=1)

    # engagement variance: 1 if engaged, 0 if pass
    X_eng = np.where(np.isnan(X), np.nan, np.where(X != 0, 1.0, 0.0))
    var_engagement = np.nanvar(X_eng, axis=0, ddof=1)

    # valence variance: only consider engaged votes
    X_val = np.where(X == 0, np.nan, X)
    var_valence = np.nanvar(X_val, axis=0, ddof=1)

    # choose variance based on mode
    if variance_mode == "overall":
        dispersions = var_overall
    elif variance_mode == "valence":
        dispersions = var_valence
    elif variance_mode == "engagement":
        dispersions = var_engagement
    else:
        raise ValueError(f"Unknown variance_mode: {variance_mode}")

    valid = coverage >= 2  # same as before

    # ---- 3. binning ---------------------------------------------------
    if n_bins is None or n_bins <= 1:
        bin_idx = np.zeros(n_statements, dtype=int)
    else:
        if bin_by == "coverage":
            bin_idx = pd.cut(coverage, bins=n_bins, labels=False)
        elif bin_by == "mean_valence":
            bin_idx = pd.cut(mean_valence, bins=n_bins, labels=False)
        elif bin_by == "mean_abs_valence":
            bin_idx = pd.cut(mean_abs_valence, bins=n_bins, labels=False)
        elif bin_by == "p_engaged":
            bin_idx = pd.cut(p_engaged, bins=n_bins, labels=False)
        else:
            raise ValueError(f"Unknown bin_by: {bin_by}")

    # ---- 4. normalize within bins ------------------------------------
    dispersions_norm = np.full(n_statements, np.nan)
    for b in np.unique(bin_idx[valid]):
        mask = (bin_idx == b) & valid
        if mask.sum() < 2:
            continue
        d = dispersions[mask]
        mu = d.mean()
        sd = d.std()
        if sd == 0 or not np.isfinite(sd):
            continue
        dispersions_norm[mask] = (d - mu) / sd

    # ---- 5. stats table ----------------------------------------------
    stats = pd.DataFrame(
        {
            "coverage": coverage,
            "mean_valence": mean_valence,
            "mean_abs_valence": mean_abs_valence,
            "p_engaged": p_engaged,
            "bin_idx": bin_idx,
            "var_overall": var_overall,
            "var_valence": var_valence,
            "var_engagement": var_engagement,
            "dispersions": dispersions,
            "dispersions_norm": dispersions_norm,
        },
        index=adata.var_names,
    )

    # ---- 6. selection -------------------------------------------------
    if n_top_statements is not None:
        # rank by normalized dispersion first, then raw
        order = np.lexsort(
            (-stats["dispersions"].values, -stats["dispersions_norm"].values)
        )
        hv = np.zeros(n_statements, dtype=bool)
        hv[order[:n_top_statements]] = True
    else:
        hv = valid.copy()
        if min_cov is not None:
            hv &= stats["coverage"].values >= min_cov
        if max_cov is not None:
            hv &= stats["coverage"].values <= max_cov
        if min_disp is not None:
            hv &= stats["dispersions_norm"].values >= min_disp
        if max_disp is not None:
            hv &= stats["dispersions_norm"].values <= max_disp

    stats[key_added] = hv

    # ---- 7. output ----------------------------------------------------
    if not inplace:
        return stats

    for k in stats.columns:
        adata.var[k] = stats[k].values

    # store metadata in .uns
    adata.uns[key_added] = {
        "variance_mode": variance_mode,
        "bin_by": bin_by,
        "n_bins": n_bins,
        "min_disp": min_disp,
        "max_disp": max_disp,
        "min_cov": min_cov,
        "max_cov": max_cov,
        "n_top_statements": n_top_statements,
        "subset": subset,
        "valid": valid,
        "statement_names": adata.var_names.tolist(),
    }

    if subset:
        adata._inplace_subset_var(hv)
