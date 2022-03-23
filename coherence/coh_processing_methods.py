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
from typing import Optional, Union

import numpy as np
import pandas as pd

from coh_exceptions import (
    DuplicateEntryError,
    MissingEntryError,
    MissingFileExtensionError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_entries import (
    check_duplicates_list,
    check_master_entries_in_sublists,
    check_sublist_entries_in_master,
)
from coh_handle_files import check_ftype_present, identify_ftype
from coh_saving import check_before_overwrite


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
    def _sort_inputs(self):
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing."""

    def _check_vars_present(
        self, master_list: list[str], sublists: list[list[str]]
    ) -> None:
        """Checks to make sure the variables in the variable order list are all
        present in the identical and unique variable lists and that the
        identical and unique variable lists are specified in the variable
        order list.

        PARAMETERS
        ----------
        master_list : list[str]
        -   A master list of values. Here the variable order list.

        sublists : list[list[str]]
        -   A list of sublists of values. Here the identical and unique
            variable lists.

        RAISES
        ------
        MissingEntryError
        -   Raised if any values in the master list or sublists are missing from
            one another.
        """

        all_present, absent_entries = check_master_entries_in_sublists(
            master_list=master_list, sublists=sublists, allow_duplicates=False
        )
        if not all_present:
            raise MissingEntryError(
                "Error when trying to convert the results of the analysis into "
                f"a DataFrame:\nThe following columns {absent_entries} do not "
                "have any data."
            )

        all_present, absent_entries = check_sublist_entries_in_master(
            master_list=master_list, sublists=sublists, allow_duplicates=False
        )
        if not all_present:
            raise MissingEntryError(
                "Error when trying to convert the results of the analysis into "
                f"a DataFrame:\nThe following columns {absent_entries} have "
                "not been accounted for when ordering the columns of the "
                "DataFrame."
            )

    def _set_df_identical_vars(
        self, var_names: list[str], var_values: dict, n_entries: int
    ) -> dict:
        """Sets the variables which have identical values regardless of the
        channel from which the data is coming.

        PARAMETERS
        ----------
        var_names : list[str]
        -   Names of the variables with identical values.

        var_values : dict
        -   Dictionary where the keys are the variable names and the values are
            the values of the variables which are identical across channels.

        n_entries : int
        -   The number of entry copies to include within each key.

        RETURNS
        -------
        dict
        -   Dictionary of key:value pairs where the keys are the variable names
            and the values a list of identical entries for the corresponding
            key.
        """

        return {name: [var_values[name]] * n_entries for name in var_names}

    @abstractmethod
    def _set_df_unique_vars(self):
        """Sets the variables which have unique values depending on the
        channel from which the data is coming."""

    def _combine_df_vars(self, identical_vars: dict, unique_vars: dict) -> dict:
        """Combines identical and unique variables together into a single
        dictionary.

        PARAMETERS
        ----------
        identical_vars : dict
        -   Dictionary in which the keys are the names of the variables whose
            values are identical across channels, and the values the variables'
            corresponding values.

        unique_vars : dict
        -   Dictionary in which the keys are the names of the variables whose
            values are different across channels, and the values the variables'
            corresponding values.

        RETURNS
        -------
        combined_vars : dict
        -   Dictionary containing the identical and unique variables.

        RAISES
        ------
        DuplicateEntryError
        -   Raised if a variable is listed multiple times within the identical
            and/or unique variables.
        """

        combined_vars = identical_vars | unique_vars

        duplicates, duplicate_values = check_duplicates_list(
            values=combined_vars.keys()
        )
        if duplicates:
            raise DuplicateEntryError(
                "Error when converting the Morlet power analysis results into "
                f"a DataFrame:\nThe DataFrame column(s) {duplicate_values} are "
                "repeated."
            )

        return combined_vars

    def _to_dataframe(
        self,
        var_order: list[str],
        identical_vars: dict,
        unique_vars: dict,
    ) -> pd.DataFrame:
        """Converts the processed data into a pandas DataFrame.

        PARAMETERS
        ----------
        var_order : list[str]
        -   The order the variables should take in the DataFrame.

        identical_vars : dict
        -   The names of the variables whose values do not depend on the
            channel, and their corresponding values.

        unique_vars : dict
        -   The names of the variables whose values depend on the channel, and
            their corresponding values.
        """

        combined_vars = self._combine_df_vars(identical_vars, unique_vars)

        return pd.DataFrame.from_dict(combined_vars, orient="columns").reindex(
            columns=var_order
        )

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""

    def _prepare_results_for_saving(
        self,
        results: np.array,
        results_structure: Optional[list[str]],
        rearrange: Optional[list[str]],
    ) -> list:
        """Copies analysis results and rearranges their dimensions as specified
        in preparation for saving.

        PARAMETERS
        ----------
        results : numpy array
        -   The results of the analysis.

        results_structure : list[str] | None; default None
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
                [results_structure.index(axis) for axis in rearrange],
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
        ftype: Optional[str],
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
        UnidencitcalEntryError
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
