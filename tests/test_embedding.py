import pytest

from valency_anndata.viz._embedding import (
    _expand_color_specs,
    _parse_color_spec,
)


class TestParseColorSpec:
    def test_single_index_parse(self):
        assert _parse_color_spec("X_pca[2]") == ("X_pca", 2, None)

    def test_non_x_key_parse(self):
        assert _parse_color_spec("evoc_polis2[1]") == ("evoc_polis2", 1, None)

    def test_plain_key_returns_none(self):
        assert _parse_color_spec("cluster") is None


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

    def test_non_x_slice_expansion(self):
        assert _expand_color_specs("evoc_polis2[1:3]") == [
            "evoc_polis2[1]",
            "evoc_polis2[2]",
        ]

    def test_mixed_list_preserves_plain_obs_keys(self):
        assert _expand_color_specs(["kmeans_pacmap", "X_pca[2:4]", "pct_seen"]) == [
            "kmeans_pacmap",
            "X_pca[2]",
            "X_pca[3]",
            "pct_seen",
        ]

    def test_mixed_list_with_non_x_key(self):
        assert _expand_color_specs(["cluster", "evoc_polis2[0]"]) == [
            "cluster",
            "evoc_polis2[0]",
        ]

    def test_malformed_syntax_rejected(self):
        with pytest.raises(ValueError, match="Invalid embedding color spec"):
            _expand_color_specs("X_pca[nope]")

    @pytest.mark.parametrize(
        "color",
        [
            "X_pca[2:]",
            "X_pca[:]",
            "X_pca[2:2]",
            "X_pca[6:2]",
        ],
    )
    def test_invalid_slices_rejected(self, color):
        with pytest.raises(ValueError, match="Invalid embedding color slice"):
            _expand_color_specs(color)
