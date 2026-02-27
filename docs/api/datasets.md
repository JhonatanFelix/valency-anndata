## Reference Datasets

These datasets are provided as a starting point for exploration and experimentation.

### ::: valency_anndata.datasets.aufstehen

### ::: valency_anndata.datasets.chile_protest

### Overview

| Dataset | Fingerprint | Participants | Statements | Completeness |
|---------|:-----------:|:------------:|:----------:|:------------:|
{% for d in reference_datasets.datasets -%}
| [{{ d.title or d.id }}]({{ d.source_url }}) | <img src="{{ d.fingerprint | replace('docs/', '../') }}" width="80"> | {{ "{:,}".format(d.participants.kept) }} / {{ "{:,}".format(d.participants.total) }} | {{ d.statements.kept }} / {{ d.statements.total }} | {{ d.matrix_completeness[-1].completeness }}% |
{% endfor %}

## Polis

### ::: valency_anndata.datasets.polis.load

### ::: valency_anndata.datasets.polis.export_csv

### ::: valency_anndata.datasets.polis.translate_statements