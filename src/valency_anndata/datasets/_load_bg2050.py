from typing import Optional
import valency_anndata as val


def bg2050(
    translate_to: Optional[str] = None,
):
    """
    Polis conversation from the BG 2050 community visioning project.

    A 33-day digital engagement where nearly 7,900 residents of Bowling Green
    and Warren County, Kentucky, shared ideas for the region's future. The
    project was commissioned by Warren County government in response to
    projections that the county will nearly double in size over 25 years, and
    was executed by Innovation Engine in partnership with The Computational
    Democracy Project and Google's Jigsaw.

    See: <https://whatcouldbgbe.com/about-the-project>

    Parameters
    ----------
    translate_to : str or None, optional
        Target language code (e.g., ``"en"``, ``"fr"``) for translating
        statement text. Defaults to None (no translation).

    Returns
    -------
    adata : anndata.AnnData
        AnnData object containing the loaded Polis conversation.

    Examples
    --------
    Load the BG 2050 conversation:

    ```py
    adata = val.datasets.bg2050()
    ```

    Load translated to French:

    ```py
    adata = val.datasets.bg2050(translate_to="fr")
    ```

    Attribution
    -----------

    Data was gathered using the Polis software (see:
    <https://compdemocracy.org/polis> and
    <https://github.com/compdemocracy/polis>) and is sub-licensed under CC BY
    4.0 with Attribution to The Computational Democracy Project. The data and
    more information about how the data was collected can be found at the
    following link: <https://pol.is/report/r7wehfsmutrwndviddnii>
    """
    adata = val.datasets.polis.load("https://pol.is/report/r7wehfsmutrwndviddnii", translate_to=translate_to)

    return adata
