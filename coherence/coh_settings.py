"""Methods for handling information in the data and analysis settings files.

CLASSES
-------
ExtractMetadata
-   Collects metadata information about the data into a dictionary.
"""


from coh_exceptions import MissingAttributeError
from coh_handle_files import load_file


def extract_metadata(
    settings: dict,
    info_keys: list[str] = [
        "cohort",
        "sub",
        "med",
        "stim",
        "ses",
        "task",
        "run",
    ],
    missing_key_error: bool = True,
) -> dict:
    """Extracts metadata information from a settings dictionary into a
    dictionary of key:value pairs corresponding to the various metadata aspects
    of the data.

    PARAMETERS
    ----------
    settings : dict
    -   Dictionary of key:value pairs containing information about the data
        being analysed.

    info_keys : list[str], optional
    -   List of strings containing the keys in the settings dictionary that
        should be extracted into the metadata information dictionary.

    missing_key_error : bool, optional
    -   Whether or not an error should be raised if a key in info_keys is
        missing from the settings dictionary. Default True. If False, None is
        given as the value of that key in the metadata information dictionary.

    RETURNS
    -------
    metadata : dict
    -   Extracted metadata.

    RAISES
    ------
    MissingAttributeError
    -   Raised if a metadata information key is missing from the settings
        dictionary and 'missing_key_error' is 'True'.
    """

    metadata = {}
    for key in info_keys:
        if key in settings.keys():
            metadata[key] = settings[key]
        else:
            if missing_key_error:
                raise MissingAttributeError(
                    f"The metadata information '{key}' is not present in "
                    "the settings file, and thus cannot be extracted."
                )
            metadata[key] = None

    return metadata


def get_analysis_settings(
    settings_fpath: str,
) -> tuple[dict, dict, list[str], list[str]]:
    """Gets the default settings for results analysis, as well as those specific
    for the requested analysis.

    PARAMETERS
    ----------
    settings_fpath : str
    -   Filepath to the analysis-specific settings.

    RETURNS
    -------
    analysis_settings : dict
    -   The analysis-specific settings.

    extract_from_dicts : dict[list[str]]
    -   The entries of dictionaries within the results to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the results dictionary.

    identical_entries : list[str]
    -   The entries in the results which are identical across channels and for
        which only one copy is present.

    discard_entries : list[str]
    -   The entries in the results which should be discarded immediately without
        processing.
    """

    analysis_settings = load_file(settings_fpath)

    extract_from_dicts = {
        "metadata": ["sub", "med", "stim", "ses", "task", "run"]
    }

    identical_entries = ["freqs"]
    discard_entries = [
        "samp_freq",
        "subject_info",
    ]

    return (
        analysis_settings,
        extract_from_dicts,
        identical_entries,
        discard_entries,
    )
