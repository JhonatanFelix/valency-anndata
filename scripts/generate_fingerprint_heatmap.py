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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Polis report URL, conversation URL, or bare ID")
    parser.add_argument("output", nargs="?", help="Output path (default: fingerprint_<id>.png)")
    parser.add_argument("--min-votes", type=int, default=7, metavar="N", help="Minimum votes per participant to include (default: 7)")
    parser.add_argument("--exclude-unvoted-statements", action="store_true", help="Drop statement columns with <1%% completion (e.g. immediately moderated out)")
    parser.add_argument("--open", action="store_true", help="Open the image in the browser after saving")
    args = parser.parse_args()

    print(f"Loading from: {args.source}")
    adata = val.datasets.polis.load(args.source)

    X = np.array(adata.X, dtype=float)

    votes_per_participant = np.sum(~np.isnan(X), axis=1)
    X = X[votes_per_participant >= args.min_votes]
    print(f"Kept {X.shape[0]} of {len(votes_per_participant)} participants (≥{args.min_votes} votes)")

    if args.exclude_unvoted_statements:
        col_completion = (~np.isnan(X)).mean(axis=0)
        X = X[:, col_completion >= 0.01]
        print(f"Kept {X.shape[1]} of {len(col_completion)} statements (≥1% completion)")

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
