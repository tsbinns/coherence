"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""




import pickle

from abc import ABC, abstractmethod
from typing import Any

from coh_saving import SaveObject, check_before_overwrite




class ProcMethod(ABC):
    """Abstract class for implementing data processing methods.
    -   This class should not be called directly. Instead, its subclasses should
        be called from their respective files.

    METHODS
    -------
    process (abstract)
    -   Performs the processing on the data.

    save (abstract)
    -   Saves the processed data to a specified location as a specified
        filetype.

    SUBCLASSES
    ----------
    PowerMorlet
    -   Performs power analysis on the data using Morlet wavelets.

    ConnectivityCoh
    -   Performs connectivity analysis on the data using coherence as the
        measure.

    ConnectivityiCoh
    -   Performs connectivity analysis on the data using the imaginary part of
        coherence as the measure.
    """

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""


    @abstractmethod
    def save(self,
        fpath: str,
        obj: Any,
        attr_to_save: list[str],
        ask_before_overwrite: bool = True
        ) -> None:
        """Saves the processed data to a specified location.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath for where to save the object.

        obj : Any
        -   The object to save.

        attr_to_save : list[str]
        -   The names of the attributes of the object to save.

        ask_before_overwrite : bool | Optional, default True
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists. If False, the user is not asked to
            confirm this and it is done automatically.
        """

        if ask_before_overwrite:
            write = check_before_overwrite(fpath, ask_before_overwrite)
        else:
            write = True

        if write:
            with open(fpath, 'wb') as file:
                pickle.dump(SaveObject(obj, attr_to_save), file)
