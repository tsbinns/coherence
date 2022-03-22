"""Methods for handling information in the data and analysis settings files.

CLASSES
-------
ExtractMetadata
-   Collects metadata information about the data into a dictionary.
"""


from coh_exceptions import MissingAttributeError


class ExtractMetadata:
    """
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
    """

    def __init__(
        self,
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
    ) -> None:

        self.settings = settings
        self.info_keys = info_keys
        self._missing_key_error = missing_key_error

        self.metadata = {}

        self._extract_metadata()

    def _extract_metadata(self) -> None:
        """Extracts metadata information from a settings dictionary into a
        dictionary of key:value pairs corresponding to the various metadata
        aspects of the data.

        RAISES
        ------
        MissingAttributeError
        -   Raised if a metadata information key is missing from the settings
            dictionary. Only raised if missing_key_error was set to True when
            the object was instantiated.
        """

        for key in self.info_keys:
            if key in self.settings.keys():
                self.metadata[key] = self.settings[key]
            else:
                if self._missing_key_error:
                    raise MissingAttributeError(
                        f"The metadata information '{key}' is not present in "
                        "the settings file, and thus cannot be extracted."
                    )
                self.metadata[key] = None
