# Real Polis export fixtures

CSV exports downloaded from https://pol.is/report/r2dxjrdwef2ybx2w9n3ja
via `polis-client` (`PolisClient.get_export_file`).

## Files

| File | Description |
|------|-------------|
| `votes.csv` | Raw vote events (voter × statement × timestamp) |
| `comments.csv` | Statements/comments with metadata. Note: does **not** include `is-seed` or `is-meta` columns — the code handles this gracefully (sets to `pd.NA`). |
| `summary.csv` | Human-readable report summary |
| `participant-votes.csv` | Per-participant vote counts and group assignment |
| `comment-groups.csv` | Per-comment vote breakdowns by consensus group |

## Refreshing

```python
from polis_client import PolisClient

client = PolisClient(base_url="https://pol.is")
for filename in ["votes.csv", "comments.csv", "summary.csv", "participant-votes.csv", "comment-groups.csv"]:
    data = client.get_export_file(report_id="r2dxjrdwef2ybx2w9n3ja", filename=filename)
    with open(filename, "w") as f:
        f.write(data)
```
