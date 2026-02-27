# /// script
# dependencies = ["valency-anndata", "matplotlib"]
# ///
"""
Generate a fingerprint heatmap for a Polis conversation or report.

The heatmap shows the participant × statement vote matrix as a colored grid
(red = disagree, green = agree, white = not seen). No axes or labels.

Usage:
    uv run scripts/generate_fingerprint_heatmap.py <source> [output_path]

Examples:
    uv run scripts/generate_fingerprint_heatmap.py "https://pol.is/report/r2dfw8eambusb8buvecjt"
    uv run scripts/generate_fingerprint_heatmap.py "6rphtwwfn4" fingerprint.png
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import valency_anndata as val


def dark_rdylgn_cmap():
    """Return RdYlGn with a darker yellow midpoint for contrast against white NaN."""
    base = plt.cm.RdYlGn
    colors = base(np.linspace(0, 1, 256))

    # Darken the central yellow region (~middle 25%) so it reads as clearly
    # distinct from the white NaN background.
    center = 128
    half_width = 32
    for i in range(center - half_width, center + half_width + 1):
        colors[i, :3] = np.clip(colors[i, :3] * 0.75, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list("RdYlGn_darkyellow", colors)
    cmap.set_bad("white")
    return cmap


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    source = sys.argv[1]

    print(f"Loading from: {source}")
    adata = val.datasets.polis.load(source)

    X = np.array(adata.X, dtype=float)
    X_masked = np.ma.masked_where(np.isnan(X), X)

    src = adata.uns.get("source", {})
    name = src.get("report_id") or src.get("conversation_id") or "fingerprint"

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = Path(f"fingerprint_{name}.png")

    cmap = dark_rdylgn_cmap()

    n_participants, n_statements = X.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(X_masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path} ({n_participants} participants × {n_statements} statements)")


if __name__ == "__main__":
    main()
