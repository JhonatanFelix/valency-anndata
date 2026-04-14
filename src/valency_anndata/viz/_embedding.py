import re
from typing import Optional, Sequence


_COLOR_SPEC_RE = re.compile(
    r"""
    ^
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)
    \[
        \s*
        (?:
            (?P<index>\d+)
            |
            (?P<start>\d*)\s*:\s*(?P<stop>\d*)
        )
        \s*
    \]
    $
    """,
    re.VERBOSE,
)


def _parse_color_spec(color: str) -> Optional[tuple[str, int, Optional[int]]]:
    if "[" not in color and "]" not in color:
        return None

    m = _COLOR_SPEC_RE.match(color)
    if not m:
        raise ValueError(
            f"Invalid embedding color spec '{color}'. "
            "Expected 'foo[i]', 'foo[a:b]', or 'foo[:b]'."
        )

    key = m.group("key")
    index = m.group("index")

    if index is not None:
        return key, int(index), None

    start = int(m.group("start") or 0)
    stop_text = m.group("stop")

    if stop_text == "":
        raise ValueError(
            f"Invalid embedding color slice '{color}'. "
            "Expected 'foo[a:b]' or 'foo[:b]' with stop > start."
        )

    stop = int(stop_text)
    if stop <= start:
        raise ValueError(
            f"Invalid embedding color slice '{color}'. "
            "Expected 'foo[a:b]' or 'foo[:b]' with stop > start."
        )

    return key, start, stop


def _expand_color_spec(color: str):
    parsed = _parse_color_spec(color)
    if parsed is None:
        return color

    key, start, stop = parsed
    if stop is None:
        return color

    return [f"{key}[{i}]" for i in range(start, stop)]


def _expand_color_specs(color: Optional[str | Sequence[str]]):
    if color is None:
        return None

    if isinstance(color, str):
        return _expand_color_spec(color)

    expanded = []
    for item in color:
        item_expanded = _expand_color_spec(item)
        if isinstance(item_expanded, list):
            expanded.extend(item_expanded)
        else:
            expanded.append(item_expanded)

    return expanded
