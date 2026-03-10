from typing import Literal, Optional
import valency_anndata as val


CubaProtestPeriod = Literal["before_1", "before_2", "after"]

_PERIOD_URLS = {
    "before_1": "https://pol.is/report/r8vwiasjhnj9a9phdsjum",
    "before_2": "https://pol.is/report/r6x3fsvscatuwrt8kbexm",
    "after": "https://pol.is/report/r2nddcf563wm9yusa9jsx",
}


def cuba_protest(
    period: CubaProtestPeriod | str,
    translate_to: Optional[str] = None,
    **kwargs,
):
    """
    Polis conversations run around Cuba's planned 15N march (November 2021).

    The 15N march was a peaceful protest planned for November 15, 2021, but was
    suppressed by the Cuban government before it could take place. Three
    conversations were run in sequence — two before the planned march and one
    after its suppression — allowing longitudinal comparison of public opinion
    around the event.

    Parameters
    ----------
    period : str
        The conversation period to load. One of:

        - ``"before_1"`` — First conversation, before the planned march
        - ``"before_2"`` — Second conversation, before the planned march
        - ``"after"`` — Conversation run after the march was suppressed

    translate_to : str or None, optional
        Target language code (e.g., ``"en"``, ``"fr"``) for translating
        statement text. Defaults to None (no translation).

    Returns
    -------
    adata : anndata.AnnData
        AnnData object containing the loaded Polis conversation.

    Examples
    --------
    Load the post-protest conversation:

    ```py
    adata = val.datasets.cuba_protest(period="after")
    ```

    Load the first pre-protest conversation with English translation:

    ```py
    adata = val.datasets.cuba_protest(period="before_1", translate_to="en")
    ```

    Attribution
    -----------

    Data was gathered using the Polis software (see:
    <https://compdemocracy.org/polis> and
    <https://github.com/compdemocracy/polis>) and is sub-licensed under CC BY
    4.0 with Attribution to The Computational Democracy Project.
    """
    if period not in _PERIOD_URLS:
        raise ValueError(f"Unknown period {period!r}. Must be one of: {list(_PERIOD_URLS)}")
    url = _PERIOD_URLS[period]

    adata = val.datasets.polis.load(url, translate_to=translate_to, **kwargs)

    return adata
