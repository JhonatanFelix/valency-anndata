import re
from typing import Optional, Sequence


_COLOR_SPEC_RE = re.compile(
    r"""
    ^
    (?P<key>X_[A-Za-z0-9_]+) #TODO: remeber to change this, matching logic too restritive
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
    re.VERBOSE,)


def _parse_color_spec(color: str) -> Optional[tuple[str, int, Optional[int]]]:
    if "[" not in color and "]" not in color:
        return None

    m = _COLOR_SPEC_RE.match(color)
    if not m:
        raise ValueError(
            f"Invalid embedding color spec '{color}'. "
            "Expected 'X_foo[i]', 'X_foo[a:b]', or 'X_foo[:b]'."
        )

    key = m.group("key")
    index = m.group("index")

    if index is not None:
        return key, int(index), None

    start = m.group("start")
    stop = m.group("stop")

    if stop == "":
        raise ValueError(
            f"Embedding color spec '{color}' is not supported yet. "
            "Use 'X_foo[i]', 'X_foo[a:b]', or 'X_foo[:b]'."
        )

    return key, int(start or 0), int(stop)


def _expand_color_specs(color: Optional[str | Sequence[str]]): # TODO: reject explictly stop <= start and len()==0 slices
    if color is None:                                          # Maybe also avoid absurd cases? [:0]?
        return None   #! The wrapper will have to be bigger because of the outputs, change that...

    if isinstance(color, str):
        parsed = _parse_color_spec(color)
        if parsed is None:
            return color

        key, start, stop = parsed
        if stop is None:
            return color

        return [f"{key}[{i}]" for i in range(start, stop)] #! X_pca[2:2] becomes []
    expanded = []
    for item in color:
        item_expanded = _expand_color_specs(item)
        if isinstance(item_expanded, list):
            expanded.extend(item_expanded)
        else:
            expanded.append(item_expanded)

    return expanded
