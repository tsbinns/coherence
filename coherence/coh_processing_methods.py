"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""

import csv
import json
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union
import numpy as np
import coh_signal
from coh_exceptions import (
    MissingFileExtensionError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_files import check_ftype_present, identify_ftype
from coh_saving import check_before_overwrite


class ProcMethod(ABC):
    """Abstract class for implementing data processing methods.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   A preprocessed Signal object whose data will be processed.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

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
    -   Performs power analysis on data using Morlet wavelets.

    PowerFOOOF
    -   Performs power analysis on data using FOOOF.

    ConnectivityCoh
    -   Performs connectivity analysis on data with coherence.

    ConnectivityiCoh
    -   Performs connectivity analysis on data with the imaginary part of
        coherence.
    """

    @abstractmethod
    def __init__(self, signal: coh_signal.Signal, verbose: bool) -> None:

        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self.processing_steps = None
        self.extra_info = None

        # Initialises inputs of the ProcMethod object.
        self.signal = deepcopy(signal)
        self._verbose = verbose

        # Initialises aspects of the ProcMethod object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._processed = False

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""

    @abstractmethod
    def _sort_inputs(self):
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing and assigns inputs."""

        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.extra_info = deepcopy(self.signal.extra_info)

    def _prepare_results_for_saving(
        self,
        results: np.array,
        results_dims: Union[list[str], None],
        rearrange: Union[list[str], None],
    ) -> list:
        """Copies analysis results and rearranges their dimensions as specified
        in preparation for saving.

        PARAMETERS
        ----------
        results : numpy array
        -   The results of the analysis.

        results_dims : list[str] | None; default None
        -   The names of the axes in the results, used for rearranging the axes.
            If None, the data cannot be rearranged.

        rearrange : list[str] | None; default None
        -   How to rearrange the axes of the data once extracted. If given,
            'results_structure' must also be given.
        -   E.g. ["channels", "epochs", "timepoints"] would give data in the
            format channels x epochs x timepoints
        -   If None, the data is taken as is.

        RETURNS
        -------
        extracted_results : array
        -   The transformed results.
        """

        extracted_results = deepcopy(results)

        if rearrange:
            extracted_results = np.transpose(
                extracted_results,
                [results_dims.index(axis) for axis in rearrange],
            )

        return extracted_results.tolist()

    def _save_as_json(self, to_save: dict, fpath: str) -> None:
        """Saves entries in a dictionary as a json file.

        PARAMETERS
        ----------
        to_save : dict
        -   Dictionary in which the keys represent the names of the entries in
            the json file, and the values represent the corresponding values.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "w", encoding="utf8") as file:
            json.dump(to_save, file)

    def _save_as_csv(self, to_save: dict, fpath: str) -> None:
        """Saves entries in a dictionary as a csv file.

        PARAMETERS
        ----------
        to_save : dict
        -   Dictionary in which the keys represent the names of the entries in
            the csv file, and the values represent the corresponding values.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "wb") as file:
            save_file = csv.writer(file)
            save_file.writerow(to_save.keys())
            save_file.writerow(to_save.values())

    def _save_as_pkl(
        self, to_save: Union["ProcMethod", dict], fpath: str
    ) -> None:
        """Pickles and saves information in any format.

        PARAMETERS
        ----------
        to_save : ProcMethod | dict
        -   Information that will be saved.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "wb") as file:
            pickle.dump(to_save, file)

    def _save_object(
        self,
        to_save: "ProcMethod",
        fpath: str,
        ask_before_overwrite: bool,
        verbose: bool,
    ) -> None:
        """Saves the ProcMethod subclass object as a .pkl file.

        PARAMETERS
        ----------
        to_save : ProcMethod
        -   ProcMethod subclass object to save.

        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.

        verbose : bool
        -   Whether or not to print a note of the saving process.
        """

        if not check_ftype_present(fpath):
            fpath += ".pkl"

        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            self._save_as_pkl(to_save=to_save, fpath=fpath)

            if verbose:
                print(f"Saving the analysis object to:\n{fpath}")

    def _save_results(
        self,
        to_save: dict,
        fpath: str,
        ftype: Union[str, None],
        ask_before_overwrite: bool,
        verbose: bool,
    ) -> None:
        """Saves the results (power and inter-trial coherence, if applicable)
        and additional information as a file.

        PARAMETERS
        ----------
        to_save : dict
        -   The results to save.

        fpath : str
        -   Location where the data should be saved.

        ftype : str
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.

        ask_before_overwrite : bool
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.

        verbose : bool
        -   Whether or not to print a note of the saving process.

        RAISES
        ------
        UnidenticalEntryError
        -   Raised if the filetype in the filepath and the specified filetype do
            not match.

        MissingFileExtensionError
        -   Raised if no filetype is present in the filetype and one is not
            specified.

        UnavailableProcessingError
        -   Raised if the given format for saving the file is in an unsupported
            format.
        """

        if check_ftype_present(fpath) and ftype is not None:
            fpath_ftype = identify_ftype(fpath)
            if fpath_ftype != ftype:
                raise UnidenticalEntryError(
                    "Error when trying to save the results of the analysis:\n "
                    f"The filetypes in the filepath ({fpath_ftype}) and in the "
                    f"requested filetype ({ftype}) do not match."
                )
        elif check_ftype_present(fpath) and ftype is None:
            ftype = identify_ftype(fpath)
        elif not check_ftype_present(fpath) and ftype is not None:
            fpath += ftype
        else:
            raise MissingFileExtensionError(
                "Error when trying to save the results of the analysis \nNo "
                "filetype is given in the filepath and no filetype has been "
                "specified."
            )

        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            if ftype == "json":
                self._save_as_json(to_save, fpath)
            elif ftype == "csv":
                self._save_as_csv(to_save, fpath)
            elif ftype == "pkl":
                self._save_as_pkl(to_save, fpath)
            else:
                raise UnavailableProcessingError(
                    "Error when trying to save the analysis results:\nThe "
                    f"{ftype} format for saving is not supported."
                )

            if verbose:
                print(f"Saving the analysis results to:\n'{fpath}'.")
