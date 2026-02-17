# /// script
# dependencies = ["valency-anndata"]
# ///
"""
Recreate Polis CSV export format (votes.csv + comments.csv) from a dataset
loaded via the API.

Usage:
    uv run export_csv.py <source> [output_dir]

Examples:
    uv run export_csv.py 6rphtwwfn4
    uv run export_csv.py https://pol.is/report/r2dfw8eambusb8buvecjt ./exports
"""
import sys
from pathlib import Path

import valency_anndata as val


def _to_seconds(series):
    """Ensure timestamps are in seconds. Truncates ms to s (no rounding)."""
    import pandas as pd

    s = pd.to_numeric(series)
    # Heuristic: values above 1e12 are milliseconds (year ~2001+)
    if (s > 1e12).any():
        s = s // 1000
    return s.astype(int)


def export_votes_csv(adata, path):
    """Export votes in Polis CSV format: timestamp,datetime,comment-id,voter-id,vote"""
    import pandas as pd

    votes = adata.uns["votes"].copy()

    # API returns ms timestamps; CSV export uses seconds
    votes["timestamp"] = _to_seconds(votes["timestamp"])

    # Ensure canonical column order; add datetime if missing
    if "datetime" not in votes.columns:
        votes["datetime"] = pd.to_datetime(votes["timestamp"], unit="s").dt.strftime(
            "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)"
        )

    votes.sort_values(["comment-id", "voter-id"], inplace=True)

    cols = ["timestamp", "datetime", "comment-id", "voter-id", "vote"]
    # Only keep columns that exist (API source may lack some)
    cols = [c for c in cols if c in votes.columns]
    votes[cols].to_csv(path, index=False)
    print(f"Wrote {len(votes)} vote rows to {path}")


def export_comments_csv(adata, path):
    """Export statements in Polis CSV format: timestamp,datetime,comment-id,author-id,agrees,disagrees,moderated,comment-body"""
    import pandas as pd

    statements = adata.uns["statements"].copy()

    # comment-id is the index after loading; reset it to a column
    if statements.index.name == "comment-id":
        statements = statements.reset_index()

    # Compute agrees/disagrees from the vote matrix
    import numpy as np

    X = adata.X
    statements["agrees"] = (np.nansum(X == 1, axis=0)).astype(int)
    statements["disagrees"] = (np.nansum(X == -1, axis=0)).astype(int)

    # API returns ms timestamps; CSV export uses seconds
    if "timestamp" in statements.columns:
        statements["timestamp"] = _to_seconds(statements["timestamp"])

    # Add datetime if missing
    if "datetime" not in statements.columns and "timestamp" in statements.columns:
        statements["datetime"] = pd.to_datetime(
            statements["timestamp"], unit="s"
        ).dt.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)")

    cols = [
        "timestamp",
        "datetime",
        "comment-id",
        "author-id",
        "agrees",
        "disagrees",
        "moderated",
        "comment-body",
    ]
    cols = [c for c in cols if c in statements.columns]
    statements[cols].to_csv(path, index=False)
    print(f"Wrote {len(statements)} statement rows to {path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    source = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from: {source}")
    adata = val.datasets.polis.load(source)

    export_votes_csv(adata, output_dir / "votes.csv")
    export_comments_csv(adata, output_dir / "comments.csv")
    print(f"\nDone! Files written to {output_dir}/")


if __name__ == "__main__":
    main()
