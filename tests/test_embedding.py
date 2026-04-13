import pytest

from valency_anndata.viz._embedding import (
    _expand_color_specs,
    _parse_color_spec,
)

# TODO: should reject X_pca[2:2], [6:2], [2:](?)
# TODO : test non "X_" after broader the parser (not just X_pca also?)

class TestParseColorSpec:
    def test_single_index_parse(self):
        assert _parse_color_spec("X_pca[2]") == ("X_pca", 2, None)


class TestExpandColorSpecs:
    def test_bounded_slice_expansion(self):
        assert _expand_color_specs("X_pca[2:6]") == [
            "X_pca[2]",
            "X_pca[3]",
            "X_pca[4]",
            "X_pca[5]",
        ]

    def test_prefix_slice_expansion(self):
        assert _expand_color_specs("X_pca[:2]") == [
            "X_pca[0]",
            "X_pca[1]",
        ]

    def test_mixed_list_preserves_plain_obs_keys(self):
        assert _expand_color_specs(["kmeans_pacmap", "X_pca[2:4]", "pct_seen"]) == [
            "kmeans_pacmap",
            "X_pca[2]",
            "X_pca[3]",
            "pct_seen",
        ]

    def test_malformed_syntax_rejected(self):
        with pytest.raises(ValueError, match="Invalid embedding color spec"):
            _expand_color_specs("X_pca[nope]")

    def test_bare_full_slice_rejected(self):
        with pytest.raises(ValueError, match="not supported yet"):
            _expand_color_specs("X_pca[:]")
