from typing import Literal, Optional
import valency_anndata as val


AmericanAssemblyCity = Literal["bowling_green", "louisville"]

_CITY_URLS = {
    "bowling_green": "https://pol.is/report/r2xcn2cdbmrzjmmuuytdk",
    "louisville": "https://pol.is/report/r2msmbb2bm7nmjxtftayt",
}


def american_assembly(
    city: AmericanAssemblyCity | str,
    translate_to: Optional[str] = None,
    **kwargs,
):
    """
    Polis conversations run by the American Assembly in Kentucky cities.

    The American Assembly is a public affairs organization that has used Polis
    to facilitate civic dialogue. These conversations were run in Bowling Green
    and Louisville, Kentucky.

    Parameters
    ----------
    city : str
        The city conversation to load. One of:

        - ``"bowling_green"`` — Bowling Green, KY (2018)
        - ``"louisville"`` — Louisville, KY (2019)

    translate_to : str or None, optional
        Target language code (e.g., ``"en"``, ``"fr"``) for translating
        statement text. Defaults to None (no translation).

    Returns
    -------
    adata : anndata.AnnData
        AnnData object containing the loaded Polis conversation.

    Examples
    --------
    Load the Bowling Green conversation:

    ```py
    adata = val.datasets.american_assembly(city="bowling_green")
    ```

    Load the Louisville conversation translated to French:

    ```py
    adata = val.datasets.american_assembly(city="louisville", translate_to="fr")
    ```

    Attribution
    -----------

    Data was gathered using the Polis software (see:
    <https://compdemocracy.org/polis> and
    <https://github.com/compdemocracy/polis>) and is sub-licensed under CC BY
    4.0 with Attribution to The Computational Democracy Project.
    """
    if city not in _CITY_URLS:
        raise ValueError(f"Unknown city {city!r}. Must be one of: {list(_CITY_URLS)}")
    url = _CITY_URLS[city]

    adata = val.datasets.polis.load(url, translate_to=translate_to, **kwargs)

    return adata
