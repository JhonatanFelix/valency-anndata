"""Load a Polis conversation, run the standard pipeline, and export to h5ad."""

from datetime import datetime, timezone
from pathlib import Path

import valency_anndata as val

REPORT_URL = "https://pol.is/report/r29kkytnipymd3exbynkd"
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports"

adata = val.datasets.polis.load(REPORT_URL, translate_to="en")

val.tools.recipe_polis(adata)
val.preprocessing.calculate_qc_metrics(adata, inplace=True)
val.tools.pacmap(adata, layer="X_masked_imputed_mean")
val.tools.localmap(adata, layer="X_masked_imputed_mean")
val.tools.kmeans(adata, use_rep="X_pacmap", key_added="kmeans_pacmap")
val.tools.kmeans(adata, use_rep="X_localmap", key_added="kmeans_localmap")

EXPORTS_DIR.mkdir(exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
out = EXPORTS_DIR / f"polis-export-{timestamp}.h5ad"
val.write(out, adata)
print(f"Exported to {out}")
