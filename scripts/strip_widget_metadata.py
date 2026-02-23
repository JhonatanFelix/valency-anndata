#!/usr/bin/env python3
"""Strip ipywidget metadata from Jupyter notebooks so they render on GitHub.

Removes:
  - notebook-level metadata.widgets key (widget state blob)
  - application/vnd.jupyter.widget-view+json from cell output data/metadata

Outputs with only widget-view data (no other mimetypes) are dropped entirely.
"""
import json
import sys
from pathlib import Path

WIDGET_KEY = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE_KEY = "widgets"


def strip_notebook(path: Path) -> bool:
    """Strip widget metadata from a notebook. Returns True if any changes were made."""
    with path.open() as f:
        nb = json.load(f)

    changed = False

    # Remove notebook-level widget state blob
    if WIDGET_STATE_KEY in nb.get("metadata", {}):
        del nb["metadata"][WIDGET_STATE_KEY]
        changed = True

    # Process each cell
    for cell in nb.get("cells", []):
        clean_outputs = []
        for output in cell.get("outputs", []):
            data = output.get("data", {})

            if WIDGET_KEY not in data:
                clean_outputs.append(output)
                continue

            # Drop widget-view mimetype from data
            del data[WIDGET_KEY]
            changed = True

            # Drop matching metadata entry if present
            if WIDGET_KEY in output.get("metadata", {}):
                del output["metadata"][WIDGET_KEY]

            # If the output has no remaining data mimetypes, drop the whole output
            if data:
                clean_outputs.append(output)
            # else: output dropped entirely (was widget-only with no text/plain etc.)

        if len(clean_outputs) != len(cell.get("outputs", [])):
            cell["outputs"] = clean_outputs
            changed = True

    if changed:
        with path.open("w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"stripped: {path}")
    else:
        print(f"clean:    {path}")

    return changed


def main():
    paths = [Path(p) for p in sys.argv[1:]] if sys.argv[1:] else list(Path("docs/notebooks").glob("*.ipynb"))
    if not paths:
        print("No notebooks found.", file=sys.stderr)
        sys.exit(1)

    any_changed = False
    for path in paths:
        any_changed |= strip_notebook(path)

    sys.exit(0)


if __name__ == "__main__":
    main()
