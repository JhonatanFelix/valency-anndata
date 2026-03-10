from typing import Literal, Optional
import valency_anndata as val


VTaiwanTopic = Literal["uber", "airbnb", "online_alcohol", "caning"]

_TOPIC_URLS = {
    "uber": "https://pol.is/report/r32beaksmhwesyum6kaur",
    "airbnb": "https://pol.is/6sc6vt",
    "online_alcohol": "https://pol.is/6nbf3mwttd",
    "caning": "https://pol.is/report/r263pyffyjvsurzs9dc6h",
}


def vtaiwan(
    topic: VTaiwanTopic | str,
    translate_to: Optional[str] = None,
    **kwargs,
):
    """
    Polis conversations from the vTaiwan collaborative policymaking process.

    vTaiwan is a civic deliberation process initiated in 2014 by the g0v
    community in Taiwan, using Polis to gather citizen perspectives on digital
    governance and social policy issues. These conversations are in Traditional
    Chinese.

    See: <https://info.vtaiwan.tw>

    Parameters
    ----------
    topic : str
        The policy topic to load. One of:

        - ``"uber"`` — Regulation of Uber and ride-sharing services (2015)
        - ``"airbnb"`` — Regulation of Airbnb and home-sharing services (2015)
        - ``"online_alcohol"`` — Online alcohol sales regulation (2016)
        - ``"caning"`` — Caning as a criminal punishment (2017)

    translate_to : str or None, optional
        Target language code (e.g., ``"en"``, ``"fr"``) for translating
        statement text. Defaults to None (no translation).

    Returns
    -------
    adata : anndata.AnnData
        AnnData object containing the loaded Polis conversation.

    Examples
    --------
    Load the Uber conversation:

    ```py
    adata = val.datasets.vtaiwan(topic="uber")
    ```

    Load the Airbnb conversation translated to English:

    ```py
    adata = val.datasets.vtaiwan(topic="airbnb", translate_to="en")
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

    adata = val.datasets.polis.load(url, translate_to=translate_to, **kwargs)

    return adata
