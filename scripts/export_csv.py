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


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    source = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    print(f"Loading from: {source}")
    adata = val.datasets.polis.load(source)

    val.datasets.polis.export_csv(adata, str(output_dir), include_huggingface_metadata=True)
    print(f"\nDone! Files written to {output_dir}/")


if __name__ == "__main__":
    main()
