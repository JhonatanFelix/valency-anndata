from typing import Literal, Optional
import valency_anndata as val


KlimaratTopic = Literal["food_land", "mobility", "energy", "housing", "production"]

_TOPIC_URLS = {
    "food_land": "https://pol.is/report/r7vainaazccymnjcfhbef",
    "mobility": "https://pol.is/report/r5bbmenm6nt3nnmf9dpvk",
    "energy": "https://pol.is/report/r2hswm7mnmhkksjm8nhmf",
    "housing": "https://pol.is/report/r6rxwkj9fhdptf5bcax8b",
    "production": "https://pol.is/report/r6bjnn8aedfakdkmndyfb",
}


def klimarat(
    topic: KlimaratTopic | str,
    translate_to: Optional[str] = None,
):
    """
    Polis conversations from Austria's Citizens' Climate Council (Klimarat).

    The Klimarat der Bürgerinnen und Bürger was Austria's national citizens'
    assembly on climate policy, convened in 2021–2022. Polis conversations were
    run for each of five topic areas to gather public input.

    See: <https://klimarat.org/>

    Parameters
    ----------
    topic : KlimaratTopic or str
        The topic area to load. One of:

        - ``"food_land"`` — Food & Land Use
        - ``"mobility"`` — Mobility
        - ``"energy"`` — Energy
        - ``"housing"`` — Housing
        - ``"production"`` — Production & Consumption

    translate_to : str or None, optional
        Target language code (e.g., ``"en"``, ``"fr"``) for translating
        statement text. Defaults to None (no translation).

    Returns
    -------
    adata : anndata.AnnData
        AnnData object containing the loaded Polis conversation.

    Examples
    --------
    Load the Energy topic conversation:

    ```py
    adata = val.datasets.klimarat(topic="energy")
    ```

    Load the Food & Land Use topic conversation with English translation:

    ```py
    adata = val.datasets.klimarat(topic="food_land", translate_to="en")
    ```

    Attribution
    -----------

    Data was gathered using the Polis software (see:
    <https://compdemocracy.org/polis> and
    <https://github.com/compdemocracy/polis>) and is sub-licensed under CC BY
    4.0 with Attribution to The Computational Democracy Project.
    """
    if topic not in _TOPIC_URLS:
        raise ValueError(f"Unknown topic {topic!r}. Must be one of: {list(_TOPIC_URLS)}")
    url = _TOPIC_URLS[topic]

    adata = val.datasets.polis.load(url, translate_to=translate_to)

    return adata
