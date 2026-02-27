"""
Generate a fingerprint heatmap for a Polis conversation or report.

The heatmap shows the participant × statement vote matrix as a colored grid
(red = disagree, green = agree, white = not seen). No axes or labels.

Usage:
    uv run scripts/generate_fingerprint_heatmap.py <source> [output_path] [--open]

Examples:
    uv run scripts/generate_fingerprint_heatmap.py "https://pol.is/report/r2dfw8eambusb8buvecjt"
    uv run scripts/generate_fingerprint_heatmap.py "6rphtwwfn4" fingerprint.png --open
"""
import argparse
import webbrowser
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import valency_anndata as val


def dark_rdylgn_cmap():
    """Return RdYlGn with a darker yellow midpoint for contrast against white NaN."""
    base = plt.cm.RdYlGn
    colors = base(np.linspace(0, 1, 256))

    # Saturate the central yellow region (~middle 25%) so it reads as clearly
    # distinct from the white NaN background. The RdYlGn midpoint is a pale
    # near-white yellow (#FFFFBF); reducing only the blue channel makes it a
    # bright saturated yellow without darkening red/green.
    center = 128
    half_width = 32
    for i in range(center - half_width, center + half_width + 1):
        colors[i, 2] = np.clip(colors[i, 2] * 0.1, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list("RdYlGn_darkyellow", colors)
    cmap.set_bad("white")
    return cmap


def completion_threshold(value):
    """Parse --exclude-unvoted-statements value: float = fraction, int = vote count."""
    if "." in value:
        f = float(value)
        if not (0.0 < f <= 1.0):
            raise argparse.ArgumentTypeError(f"Fraction must be between 0 and 1, got {f}")
        return f
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"Vote count must be ≥ 1, got {n}")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Polis report URL, conversation URL, or bare ID")
    parser.add_argument("output", nargs="?", help="Output path (default: fingerprint_<id>.png)")
    parser.add_argument("--min-votes", type=int, default=7, metavar="N", help="Minimum votes per participant to include (default: 7)")
    parser.add_argument("--exclude-unvoted-statements", nargs="?", const=0.01, default=None,
                        type=completion_threshold, metavar="THRESHOLD",
                        help="Drop low-completion statement columns. THRESHOLD: fraction e.g. 0.05, or integer vote count e.g. 5. Default when flag is given: 0.01")
    parser.add_argument("--open", action="store_true", help="Open the image in the browser after saving")
    args = parser.parse_args()

    print(f"Loading from: {args.source}")
    adata = val.datasets.polis.load(args.source)

    X = np.array(adata.X, dtype=float)
    n_participants_total = X.shape[0]
    n_statements_total = X.shape[1]

    votes_per_participant = np.sum(~np.isnan(X), axis=1)
    X = X[votes_per_participant >= args.min_votes]
    n_participants_excluded = n_participants_total - X.shape[0]

    threshold = args.exclude_unvoted_statements
    n_statements_excluded = 0
    if threshold is not None:
        col_votes = (~np.isnan(X)).sum(axis=0)
        if isinstance(threshold, float):
            mask = (col_votes / X.shape[0]) >= threshold
            threshold_desc = f"{threshold * 100:.1f}% completion"
        else:
            mask = col_votes >= threshold
            threshold_desc = f"{threshold} votes"
        n_statements_excluded = int((~mask).sum())
        X = X[:, mask]

    print(f"Participants: {n_participants_total} total, {n_participants_excluded} excluded (< {args.min_votes} votes), {X.shape[0]} kept")
    if threshold is not None:
        print(f"Statements:  {n_statements_total} total, {n_statements_excluded} excluded (< {threshold_desc}), {X.shape[1]} kept")
    else:
        print(f"Statements:  {n_statements_total} total, none excluded, {X.shape[1]} kept")

    n_cols = X.shape[1]
    print("\nMatrix completeness by statement quartile:")
    for pct in [25, 50, 75, 100]:
        col_end = int(np.ceil(n_cols * pct / 100))
        slice_ = X[:, :col_end]
        completeness = (~np.isnan(slice_)).mean() * 100
        print(f"  first {pct:3d}% of statements (cols 0–{col_end - 1}): {completeness:.1f}% complete")
    print()

    X_masked = np.ma.masked_where(np.isnan(X), X)

    src = adata.uns.get("source", {})
    name = src.get("report_id") or src.get("conversation_id") or "fingerprint"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("exports") / "heatmaps" / f"fingerprint_{name}.png"
        output_path.parent.mkdir(exist_ok=True)

    cmap = dark_rdylgn_cmap()

    n_participants, n_statements = X.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(X_masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path} ({n_participants} participants × {n_statements} statements)")

    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
