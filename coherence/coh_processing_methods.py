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
    def _sort_inputs(self):
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing."""

    @abstractmethod
    def _update_processing_steps(self):
        """Updates the 'processing_steps' dictionary of the processing method
        object."""

    @abstractmethod
    def _check_identical_ch_orders(self):
        """Checks to make sure that the order of the channels (and thus, the
        data) in the preprocessed data and power data is identical."""

    @abstractmethod
    def _check_vars_present(self):
        """Checks to make sure the variables in the variable order list are all
        present in the identical and unique variable lists and that the
        identical and unique variable lists are specified in the variable
        order list."""

    @abstractmethod
    def _set_df_identical_vars(self):
        """Sets the variables which have identical values regardless of the
        channel from which the data is coming."""

    @abstractmethod
    def _set_df_unique_vars(self):
        """Sets the variables which have unique values depending on the
        channel from which the data is coming."""

    @abstractmethod
    def _combine_df_vars(self):
        """Combines identical and unique variables together into a single
        dictionary."""

    @abstractmethod
    def _to_dataframe(self):
        """Converts the processed data into a pandas DataFrame."""

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""

    @abstractmethod
    def save(
        self,
        fpath: str,
        obj: Any,
        attr_to_save: list[str],
        ask_before_overwrite: bool = True,
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

        ask_before_overwrite : bool; default True
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists. If False, the user is not asked to
            confirm this and it is done automatically.
        """

        if ask_before_overwrite:
            write = check_before_overwrite(fpath, ask_before_overwrite)
        else:
            write = True

        if write:
            with open(fpath, "wb") as file:
                pickle.dump(SaveObject(obj, attr_to_save), file)
