## Reference Datasets

These datasets are provided as a starting point for exploration and experimentation.

### ::: valency_anndata.datasets.aufstehen

### ::: valency_anndata.datasets.chile_protest

### Overview

<div class="datasets-overview" markdown>

| Dataset | Participants | Statements | <abbr title="Vote matrix completeness at each quartile of statements (25% / 50% / 75% / 100%), ordered by statement ID. Each value is the % of non-missing votes across all kept participants × the first N statements.">Completeness</abbr> | Fingerprint |
|---------|:------------:|:----------:|:----------------:|:-----------:|
{% for d in reference_datasets.datasets -%}
| [{{ d.title or d.id }}]({{ d.source_url }}) | {{ "{:,}".format(d.participants.kept) }} / {{ "{:,}".format(d.participants.total) }} | {{ d.statements.kept }} / {{ d.statements.total }} | {% for q in d.matrix_completeness %}{{ q.completeness | round | int }}%{% if not loop.last %} / {% endif %}{% endfor %} | <img src="{{ d.fingerprint | replace('docs/', '../../') }}" width="80"> |
{% endfor %}

</div>

## Polis

### ::: valency_anndata.datasets.polis.load

### ::: valency_anndata.datasets.polis.export_csv

### ::: valency_anndata.datasets.polis.translate_statements
