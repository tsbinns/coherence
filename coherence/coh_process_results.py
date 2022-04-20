"""Classes and methods for applying post-processing to results.

CLASSES
-------
PostProcess
-   Class for the post-processing of results derived from raw signals.

METHODS
-------
load_results_of_types
-   Loads results of a multiple types of data and merges them into a single
    PostProcess object.

load_results_of_type
-   Loads results of a single type of data and appends them into a single
    PostProcess object
"""

from copy import deepcopy
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from coh_handle_files import generate_results_fpath, load_file

from coh_exceptions import (
    DuplicateEntryError,
    EntryLengthError,
    InputTypeError,
    MissingEntryError,
    PreexistingAttributeError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_entries import check_lengths_list_identical


class PostProcess:
    """Class for the post-processing of results derived from raw signals.

    PARAMETERS
    ----------
    results : dict
    -   A dictionary containing results to process.
    -   The entries in the dictionary should be either lists, numpy arrays, or
        dictionaries.
    -   Entries which are dictionaries will have their values treated as being
        identical for all values in the 'results' dictionary, given they are
        extracted from these dictionaries into the results.

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_entries : list[str] | None; default None
    -   The entries in 'results' which are identical across channels and for
        which only one copy is present.

    discard_entries : list[str] | None; default None
    -   The entries which should be discarded immediately without processing.

    freq_bands : dict | None; default None
    -   Dictionary containing the frequency bands whose results should also be
        calculated.
    -   Each key is the name of the frequency band, and each value is a list of
        numbers representing the lower- and upper-most boundaries of the
        frequency band, respectively, in the frequency units present in the
        results.

    METHODS
    -------
    average
    -   Averages results.

    subtract
    -   Subtracts results.

    append
    -   Appends other dictionaries of results to the list of result dictionaries
        stored in the PostProcess object.

    merge
    -   Merge dictionaries of results containing different keys into the
        results.
    """

    def __init__(
        self,
        results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_entries: Optional[list[str]] = None,
        discard_entries: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> None:

        # Initialises inputs of the object.
        results = self._sort_inputs(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
        )
        self._results = self._results_to_df(results=results)
        self._verbose = verbose

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._fbands = None
        self._fband_measures = []
        self._fband_desc_measures = []
        self._fband_columns = []
        self._var_measures = []
        self._var_columns = []
        self._desc_measures = []
        self._desc_fband_measures = ["max", "min", "fmax", "fmin"]
        self._desc_var_measures = ["std", "sem"]

    def _check_input_entry_lengths(
        self, results: dict, identical_entries: Union[list[str], None]
    ) -> int:
        """Checks that the lengths of list and numpy array entries in 'results'
        have the same length of axis 0.

        PARAMETERS
        ----------
        results : dict
        -   The results whose entries will be checked.

        identical_entries : list[str] | None
        -   The entries in 'results' which are identical across channels and for
            which only one copy is present.
        -   These entries are not included when checking the lengths, as these
            will be handled later.

        RETURNS
        -------
        length : int
        -   The lenghts of the 0th axis of lists and numpy arrays in 'results'.

        RAISES
        ------
        InputTypeError
        -   Raised if the 'results' contain an entry that is neither a list,
            numpy array, or dictionary.

        EntryLengthError
        -   Raised if the list or numpy array entries in 'results' do not all
            have the same length along axis 0.
        """

        supported_dtypes = [list, NDArray, dict]
        check_len_dtypes = [list, NDArray]

        to_check = []

        for key, value in results.items():
            if key not in identical_entries:
                dtype = type(value)
                if dtype in supported_dtypes:
                    if dtype in check_len_dtypes:
                        to_check.append(value)
                else:
                    raise InputTypeError(
                        "Error when trying to process the results:\nThe "
                        f"results dictionary contains an entry ('{key}') that "
                        f"is not of a supported data type ({supported_dtypes})."
                    )

        identical, length = check_lengths_list_identical(
            to_check=to_check, axis=0
        )
        if not identical:
            raise EntryLengthError(
                "Error when trying to process the results:\nThe length of "
                "entries in the results along axis 0 is not identical, but "
                "should be."
            )

        return length

    def _sort_inputs(
        self,
        results: dict,
        extract_from_dicts: Union[dict[list[str]], None],
        identical_entries: Union[list[str], None],
        discard_entries: Union[list[str], None],
    ) -> None:
        """Checks that the values in 'results' are in the appropriate format for
        processing.

        PARAMETERS
        ----------
        results : dict
        -   The results which will be checked.
        -   Entries which are lists or numpy arrays should have the same length
            of axis 0.

        extract_from_dicts : dict[list[str]] | None
        -   The entries of dictionaries within 'results' to include in the
            processing.
        -   Entries which are extracted are treated as being identical for all
            values in the 'results' dictionary.

        identical_entries : list[str] | None
        -   The entries in 'results' which are identical across channels and for
            which only one copy is present.

        discard_entries : list[str] | None
        -   The entries which should be discarded immediately without
            processing.

        RETURNS
        -------
        dict
        -   The results with requested dictionary entries extracted to the
            results, if applicable, and the dictionaries subsequently removed.
        """

        if discard_entries is not None:
            results = self._sort_discard_entries(
                results=results, discard_entries=discard_entries
            )

        entry_length = self._check_input_entry_lengths(
            results=results, identical_entries=identical_entries
        )

        if identical_entries is not None:
            results = self._sort_identical_entries(
                results=results,
                identical_entries=identical_entries,
                entry_length=entry_length,
            )

        results = self._sort_dicts(
            results=results,
            extract_from_dicts=extract_from_dicts,
            entry_length=entry_length,
        )

        return results

    def _sort_discard_entries(
        self, results: dict, discard_entries: list[str]
    ) -> dict:
        """Drops the specified entries from 'results'.

        PARAMETERS
        ----------
        results : dict
        -   The results dictionary with entries to discard.

        discard_entries : list[str]
        -   The entries in 'results' to discard.

        RETURNS
        -------
        results : dict
        -   The sorted results with specified entries discarded.
        """

        for entry in discard_entries:
            if entry in results.keys():
                del results[entry]
            else:
                print(
                    f"The '{entry}' attribute is not present in the results "
                    "dictionary, so cannot be deleted."
                )

        return results

    def _sort_identical_entries(
        self, results: dict, identical_entries: list[str], entry_length: int
    ) -> dict:
        """Creates a list equal to the length of other entries in 'results' for
        all entries specified in 'identical_entries', where each element of the
        list is a copy of the specified entries.

        PARAMETERS
        ----------
        results : dict
        -   The results dictionary with identical entries to sort.

        identical_entries : list[str]
        -   The entries in 'results' to convert to a list with length of axis 0
            equal to that of the 0th axis of other entries.

        entry_length : int
        -   The length of the 0th axis of entries in 'results'.

        RETURNS
        -------
        results : dict
        -   The results dictionary with identical entries sorted.
        """

        for entry in identical_entries:
            results[entry] = [results[entry]] * entry_length

        return results

    def _add_dict_entries_to_results(
        self, results: dict, extract: dict[list[str]], entry_length: int
    ) -> dict:
        """Extracts entries from dictionaries in 'results' and adds them to the
        results as a list whose length matches that of the other 'results'
        entries which are lists or numpy arrays.

        PARAMETERS
        ----------
        results : dict
        -   The results containing the dictionaries whose values should be
            extracted.

        extract : dict[list[str]]
        -   Dictionary whose keys are the names of dictionaries in 'results',
            and whose values are a list of strings corresponding to the entries
            in the dictionaries in 'results' to extract.

        entry_length : int
        -   The length of the 0th axis of entries in 'results'.

        RETURNS
        -------
        results : dict
        -   The results with the desired dictionary entries extracted.
        """

        for dict_name, dict_entries in extract.items():
            for entry in dict_entries:
                to_add = deepcopy([results[dict_name][entry]] * entry_length)
                if isinstance(to_add, dict):
                    raise TypeError(
                        "Error when processing the results:\nThe results "
                        f"contain the dictionary '{dict_name}' which contains "
                        f"an entry '{entry}' that is being extracted and "
                        "included with the results for processing, however "
                        "processing dictionaries in a PostProcess object is "
                        "not supported."
                    )
                if entry in results.keys():
                    raise PreexistingAttributeError(
                        "Error when processing the results:\nThe entry "
                        f"'{entry}' from the dictionary '{dict_name}' is being "
                        "extracted and added to the results, however an "
                        f"attribute named '{entry}' is already present in the "
                        "results."
                    )
                results[entry] = to_add

        return results

    def _drop_dicts_from_results(self, results: dict) -> dict:
        """Removes dictionaries from 'results' after the requested entries, if
        applicable, have been extracted.

        PARAMETERS
        ----------
        results : dict
        -   The results with dictionaries entries to drop.

        RETURNS
        -------
        results : dict
        -   The results with dictionary entries dropped.
        """

        to_drop = []
        for key, value in results.items():
            if isinstance(value, dict):
                to_drop.append(key)

        for key in to_drop:
            del results[key]

        return results

    def _sort_dicts(
        self,
        results: dict,
        extract_from_dicts: Union[dict[list[str]], None],
        entry_length: int,
    ) -> dict:
        """Handles the presence of dictionaries within 'results', extracting the
        requested entries, if applicable, before discarding the dictionaries.

        PARAMETERS
        ----------
        results : dict
        -   The results to sort.

        extract_from_dicts : dict[list[str]] | None
        -   The entries of dictionaries within 'results' to include in the
            processing.
        -   Entries which are extracted are treated as being identical for all
            values in the 'results' dictionary.

        entry_length : int
        -   The length of the 0th axis of entries in 'results'.

        RETURNS
        -------
        dict
        -   The sorted results, with the desired dictionary entries extracted,
            if applicable, and the dictionaries discarded.
        """

        if extract_from_dicts is not None:
            results = self._add_dict_entries_to_results(
                results=results,
                extract=extract_from_dicts,
                entry_length=entry_length,
            )

        return self._drop_dicts_from_results(results=results)

    def _results_to_df(
        self,
        results: dict,
    ) -> pd.DataFrame:
        """Converts the dictionary of results into a pandas DataFrame for
        processing, adding frequency band-wise results, if requested.

        PARAMETERS
        ----------
        results : dict
        -   A dictionary containing results to process.
        -   The entries in the dictionary should be either lists, numpy arrays,
            or dictionaries.
        -   Entries which are dictionaries will have their values treated as
            being identical for all values in the 'results' dictionary.

        RETURNS
        -------
        results : pandas DataFrame
        -   The 'results' dictionary as a DataFrame.
        """

        results = pd.DataFrame.from_dict(data=results, orient="columns")

        return results

    def _check_non_duplicates(self, results: Union[dict, pd.DataFrame]) -> None:
        """Checks that the existing results and the results being added contain
        all the same keys.

        PARAMETERS
        ----------
        results : dict | pandas DataFrame
        -   The results being added.

        RAISES
        ------
        MissingEntryError
        -   Raised if the existing results and results being added do not
            contain the same keys.
        """

        current_keys = list(self._results.keys())
        new_keys = list(results.keys())

        non_duplicate_keys = [
            key for key in current_keys if key not in new_keys
        ].extend([key for key in new_keys if key not in current_keys])

        if non_duplicate_keys:
            raise MissingEntryError(
                "Error when appending results to the PostProcess object:\nThe "
                f"key(s) {non_duplicate_keys} is(are) not present in both the "
                "existing results and the results being appended. If you still "
                "wish to add these new results, the 'merge' method should be "
                "used."
            )

    def append_from_dict(
        self,
        new_results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_entries: Optional[list[str]] = None,
        discard_entries: Optional[list[str]] = None,
    ) -> None:
        """Appends a dictionary of results to the results stored in the
        PostProcess object.

        PARAMETERS
        ----------
        new_results : dict
        -   A dictionary containing results to add.
        -   The entries in the dictionary should be either lists, numpy arrays,
            or dictionaries.
        -   Entries which are dictionaries will have their values treated as
            being identical for all values in the 'results' dictionary, given
            they are extracted from these dictionaries into the results.

        extract_from_dicts : dict[list[str]] | None; default None
        -   The entries of dictionaries within 'results' to include in the
            processing.
        -   Entries which are extracted are treated as being identical for all
            values in the 'results' dictionary.

        identical_entries : list[str] | None; default None
        -   The entries in 'results' which are identical across channels and for
            which only one copy is present.

        discard_entries : list[str] | None; default None
        -   The entries which should be discarded immediately without
            processing.
        """

        new_results = self._sort_inputs(
            results=new_results,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
        )

        self._check_non_duplicates(results=new_results)

        new_results = self._results_to_df(results=new_results)

        self._results = pd.concat(
            objs=[self._results, new_results], ignore_index=True
        )

    def append_from_df(
        self,
        new_results: pd.DataFrame,
    ) -> None:
        """Appends a DataFrame of results to the results stored in the
        PostProcess object.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   The new results to append.
        """

        self._check_non_duplicates(results=new_results)

        self._results = pd.concat(
            objs=[self._results, new_results], ignore_index=True
        )

    def _make_results_mergeable(
        self, results_1: pd.DataFrame, results_2: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Converts results DataFrames into a format that can be handled by the
        pandas function 'merge' by converting any lists into tuples.

        PARAMETERS
        ----------
        results_1: pandas DataFrame
        -   The first DataFrame to make mergeable.

        results_2: pandas DataFrame
        -   The second DataFrame to make mergeable.

        RETURNS
        -------
        pandas DataFrame
        -   The first DataFrame made mergeable.

        pandas DataFrame
        -   The second DataFrame made mergeable.
        """

        dataframes = [results_1, results_2]

        for df_i, dataframe in enumerate(dataframes):
            for row_i in dataframe.index:
                for key in dataframe.keys():
                    if isinstance(dataframe[key][row_i], list):
                        dataframe[key][row_i] = tuple(dataframe[key][row_i])
            dataframes[df_i] = dataframe

        return dataframes[0], dataframes[1]

    def _restore_results_after_merge(
        self, results: pd.DataFrame
    ) -> pd.DataFrame:
        """Converts a results DataFrame into its original format after merging
        by converting any tuples back to lists.

        PARAMETERS
        ----------
        results : pandas DataFrame
        -   The DataFrame with lists to restore from tuples.

        RETURNS
        -------
        results : pandas DataFrame
        -   The restored DataFrame.
        """

        for row_i in results.index:
            for key in results.keys():
                if isinstance(results[key][row_i], tuple):
                    results[key][row_i] = list(results[key][row_i])

        return results

    def _check_missing_before_merge(
        self, results_1: pd.DataFrame, results_2: pd.DataFrame
    ) -> None:
        """Checks that merging pandas DataFrames with the 'merge' method and
        the 'how' parameter set to 'outer' will not introduce new rows into the
        merged results DataFrame, resulting in some rows having NaN values for
        columns not present in their original DataFrame, but present in the
        other DataFrames being merged.
        -   This can occur if the column names which are shared between the
            DataFrames do not have all the same entries between the DataFrames,
            leading to new rows being added to the merged DataFrame.

        PARAMETERS
        ----------
        results_1 : pandas DataFrame
        -   The first DataFrame to check.

        results_2 : pandas DataFrame
        -   The second DataFrame to check.

        RAISES
        ------
        MissingEntryError
        -   Raised if the DataFrames' shared columns do not have values that are
            identical in the other DataFrame, leading to rows being excluded
            from the merged DataFrame.
        """

        if len(results_1.index) == len(results_2.index):
            test_merge = pd.merge(results_1, results_2, how="inner")
            if len(test_merge.index) != len(results_1.index):
                raise EntryLengthError(
                    "Error when trying to merge two sets of results with "
                    "'allow_missing' set to 'False':\nThe shared columns of "
                    "the DataFrames being merged do not have identical values "
                    "in the other DataFrame, leading to "
                    f"{len(test_merge.index)-len(results_1.index)} new row(s) "
                    "being included in the merged DataFrame.\nIf you still "
                    "want to merge these results, set 'allow_missing' to "
                    "'True'."
                )
        else:
            raise EntryLengthError(
                "Error when trying to merge two sets of results with "
                "'allow_missing' set to 'False':\nThere is an unequal number "
                "of channels present in the two sets of results being merged "
                f"({len(results_1.index)} and {len(results_2.index)}). Merging "
                "these results will lead to some attributes of the results "
                "having NaN values.\nIf you still want to merge these results, "
                "set 'allow_missing' to 'True'."
            )

    def _check_keys_before_merge(self, new_results: pd.DataFrame) -> None:
        """Checks that the column names in the DataFrames being merged are not
        identical.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   The new results being added.

        RAISES
        ------
        DuplicateEntryError
        -   Raised if there are no columns that are unique to the DataFrames
            being merged.
        """

        duplicates = []
        for key in new_results.keys():
            if key in self._results.keys():
                duplicates.append(True)
            else:
                duplicates.append(False)

        if all(duplicates):
            raise DuplicateEntryError(
                "Error when trying to merge results:\nThere are no new columns "
                "in the results being added. If you still want to add the "
                "results, use the append methods."
            )

    def merge_from_dict(
        self,
        new_results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_entries: Optional[list[str]] = None,
        discard_entries: Optional[list[str]] = None,
        allow_missing: bool = False,
    ) -> None:
        """Merges a dictionary of results to the results stored in the
        PostProcess object.

        PARAMETERS
        ----------
        new_results : dict
        -   A dictionary containing results to add.
        -   The entries in the dictionary should be either lists, numpy arrays,
            or dictionaries.
        -   Entries which are dictionaries will have their values treated as
            being identical for all values in the 'results' dictionary, given
            they are extracted from these dictionaries into the results.

        extract_from_dicts : dict[list[str]] | None; default None
        -   The entries of dictionaries within 'results' to include in the
            processing.
        -   Entries which are extracted are treated as being identical for all
            values in the 'results' dictionary.

        identical_entries : list[str] | None; default None
        -   The entries in 'results' which are identical across channels and for
            which only one copy is present.

        discard_entries : list[str] | None; default None
        -   The entries which should be discarded immediately without
            processing.

        allow_missing : bool; default False
        -   Whether or not to allow new rows to be present in the merged results
            with NaN values for columns not shared between the results being
            merged if the shared columns do not have matching values.
        -   I.e. if you want to make sure you are merging results from the same
            channels, set this to False, otherwise results from different
            channels will be merged and any missing information will be set to
            NaN.
        """

        new_results = self._sort_inputs(
            results=new_results,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
        )

        new_results = self._results_to_df(results=new_results)

        self._check_keys_before_merge(new_results=new_results)

        current_results, new_results = self._make_results_mergeable(
            results_1=self._results, results_2=new_results
        )

        if not allow_missing:
            self._check_missing_before_merge(
                results_1=current_results, results_2=new_results
            )

        merged_results = pd.merge(current_results, new_results, how="outer")

        self._results = self._restore_results_after_merge(
            results=merged_results
        )

    def merge_from_df(
        self,
        new_results: pd.DataFrame,
        allow_missing: bool = False,
    ) -> None:
        """Merges a dictionary of results to the results stored in the
        PostProcess object.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   A DataFrame containing results to add.

        allow_missing : bool; default False
        -   Whether or not to allow new rows to be present in the merged results
            with NaN values for columns not shared between the results being
            merged if the shared columns do not have matching values.
        -   I.e. if you want to make sure you are merging results from the same
            channels, set this to False, otherwise results from different
            channels will be merged and any missing information will be set to
            NaN.
        """

        self._check_keys_before_merge(new_results=new_results)

        current_results, new_results = self._make_results_mergeable(
            results_1=self._results, results_2=new_results
        )

        if not allow_missing:
            self._check_missing_before_merge(
                results_1=current_results, results_2=new_results
            )

        merged_results = pd.merge(current_results, new_results, how="outer")

        self._results = self._restore_results_after_merge(
            results=merged_results
        )

    def as_df(self) -> pd.DataFrame:
        """Returns the results as a pandas DataFrame.

        RETURNS
        -------
        results : pandas DataFrame
        -   The results as a pandas DataFrame.
        """

        return self._results

    def _check_identical_keys(
        self, indices: list[list[int]], keys: list[str]
    ) -> None:
        """Checks that the values for keys marked as identical are.

        PARAMETERS
        ----------
        indices : list[list[int]]
        -   The indices whose values should be checked.
        -   Each entry is a list of integers corresponding to the indices of
            the results to compare.

        keys : list[str]
        -   The names of the attributes of the data whose values should be
            compared

        RAISES
        ------
        UnidenticalEntryError
        -   Raised if the values being compared for any of the keys and indices
            are not identical.
        """

        for key in keys:
            for idcs in indices:
                if len(idcs) > 1:
                    for entry_i, idx in enumerate(idcs):
                        if entry_i == 0:
                            compare_against = self._results[key][idx]
                        else:
                            if self._results[key][idx] != compare_against:
                                raise UnidenticalEntryError(
                                    "Error when checking that the attributes "
                                    "of results belonging to the same group "
                                    "share the same values:\nThe values of "
                                    f"'{key}' in rows {idcs[0]} and {idx} do "
                                    "not match."
                                )

    def _get_indices_to_process(
        self,
        unique_entry_idcs: list[list[int]],
        over_key: str,
        over_entries: list,
    ) -> list[list[int]]:
        """Gets the unique indices of nodes in the results that should be
        processed.

        PARAMETERS
        ----------
        unique_entry_idcs : list[list[int]]
        -   Indices of each unique set of nodes in the results.

        over_key : str
        -   Name of the attribute in the results to process over.

        over_entries : list
        -   The values of the 'over_key' attribute in the results to process.
        -   Indices without 'over_entries' values in the 'over_key' attribute
            of the results are not included in the indices to process.

        RETURNS
        -------
        entry_idcs_to_process : list[list[int]]
        -   Unique indices of nodes in the results that should be processed.
        """

        entry_idcs_to_process = []
        for idcs in unique_entry_idcs:
            process_idcs = [
                idx
                for idx in idcs
                if self._results[over_key][idx] in over_entries
            ]
            if process_idcs != []:
                entry_idcs_to_process.append(process_idcs)

        return entry_idcs_to_process

    def _get_combined_entries(self, group_keys: list[str]) -> list[str]:
        """Combines the values of the data for each node.

        PARAMETERS
        ----------
        group_keys : list[str]
        -   Names of the attributes in the results whose values will be
            combined.

        RETURNS
        -------
        combined_entries : list[str]
        -   Combined values of all requested attributes in the results, with
            each entry corresponding to the values of each node in the results.
        """

        combined_entries = []
        for row_i in range(len(self._results.index)):
            combined_entries.append("")
            for key in group_keys:
                combined_entries[row_i] += str(self._results[key][row_i])

        return combined_entries

    def _get_unique_indices(
        self, combined_entries: list[str]
    ) -> list[list[int]]:
        """Gets the indices of each unique set of nodes in the results,
        corresponding to the sets of results that will be processed together.

        PARAMETERS
        ----------
        combined_entries : list[str]
        -   Combined values of attributes in the results, with
            each entry corresponding to the values of each node in the results.

        RETURNS
        -------
        unique_entry_idcs : list[list[int]]
        -   Indices of each unique set of nodes in the results.
        """

        unique_entries = np.unique(combined_entries).tolist()
        unique_entry_idcs = []
        for unique_entry in unique_entries:
            unique_entry_idcs.append([])
            for entry_i, combined_entry in enumerate(combined_entries):
                if unique_entry == combined_entry:
                    unique_entry_idcs[-1].append(entry_i)

        return unique_entry_idcs

    def _populate_columns(
        self,
        attributes: list[str],
        fill: Optional[Any] = None,
    ) -> None:
        """Creates placeholder columns to add to the results DataFrame.

        PARAMETERS
        ----------
        attributes : list[str]
        -   Names of the columns to add.

        fill : Any; default None
        -   Placeholder values in the columns.
        """

        for attribute in attributes:
            self._results[attribute] = [deepcopy(fill)] * len(
                self._results.index
            )

    @property
    def attributes(self) -> list[str]:
        """Gets the original attributes of the results, not those which are
        related to frequency band-wise or variability results.

        RETURNS
        -------
        list[str]
        -   Names of the results attributes not related to frequency band-wise
            or variability results.
        """

        keys = self._results.keys().tolist()
        exclude_keys = [*self._fband_columns, *self._var_columns]

        return [key for key in keys if key not in exclude_keys]

    def _get_process_keys(self, process_keys: list[str]) -> list[str]:
        """Gets the attributes of the results to process, adding and frequency
        band- and variabilty-related attributes.

        PARAMETERS
        ----------
        process_keys : list[str]
        -   Attributes of the results to process, not including those generated
            as part of frequency band-wise analyses and variability measures.

        RETURNS
        -------
        process_keys : list[str]
        -   Attributes of the results to process.
        """

        process_keys = deepcopy(process_keys)

        if self._fbands:
            fband_keys = []
            for key in process_keys:
                fband_keys.extend(
                    [
                        f"{key}_fbands_{measure}"
                        for measure in self._fband_measures
                        if measure not in self._desc_measures
                    ]
                )
            process_keys.extend(fband_keys)

        return process_keys

    def _prepare_fband_results(
        self, bands: dict, attributes: list[str], measures: list[str]
    ) -> None:
        """Checks that the inputs for calculating frequency band-wise results
        are appropriate, finds the indices of the frequency band bounds in
        the results, and adds placeholder columns that will be filled with
        frequency band results if none are already present.

        PARAMETERS
        ----------
        bands : dict
        -   Dictionary containing the frequency bands whose results should also
            be calculated.
        -   Each key is the name of the frequency band, and each value is a list
            of numbers representing the lower- and upper-most boundaries of the
            frequency band, respectively, in the frequency units present in the
            results.

        attributes : list[str]
        -   Attributes of the results to apply the frequency band-wise analysis
            to.

        measures : list[str]
        -   Measures to compute for the frequency bands.
        -   Supported inputs are: "average" for the average value; "median" for
            the median value; "max" for the maximum value; "fmax" for the
            frequency at which the maximum value occurs; "min" for the minimum
            value; and "fmin" for the frequency at which the minimum value
            occurs.

        RAISES
        ------
        EntryLengthError
        -   Raised if a frequency band does not consist of two values (i.e. a
            lower- and upper-bound for the band).

        ValueError
        -   Raised if a frequency band contains an upper- or lower-bound of
            frequencies that is not present in the results. Units of the band
            limits are assumed to be the same as in the results.
        """

        for name, freqs in bands.items():
            if len(freqs) != 2:
                raise EntryLengthError(
                    "Error when trying to compute the frequency band-wise "
                    f"results:\nThe frequency band '{name}' does not have the "
                    f"required two frequency values, but is instead {freqs}."
                )
            for freq in freqs:
                if freq not in self._results["freqs"]:
                    raise ValueError(
                        "Error when trying to compute the frequency "
                        "band-wise results:\nThe frequencies in the range "
                        f"{freqs[0]} - {freqs[1]} (units identical to those in "
                        "the results) are not present in the results with "
                        f"frequency range {self._results['freqs'][0]} - "
                        f"{self._results['freqs'][1]}."
                    )

        supported_measures = [
            "average",
            "median",
            "max",
            "min",
            "fmax",
            "fmin",
        ]
        for measure in measures:
            if measure not in supported_measures:
                raise ValueError(
                    "Error when trying to compute the frequency band-wise "
                    f"results:\nThe measure '{measure}' is not recognised. "
                    f"Supported measures are: {supported_measures}"
                )

        if not self._fbands:
            fband_columns = ["fband_labels", "fband_freqs"]
            for attribute in attributes:
                for measure in measures:
                    fband_columns.append(f"{attribute}_fbands_{measure}")
            self._populate_columns(attributes=fband_columns)
            self._fband_columns = fband_columns

    def _get_band_freq_indices(self, bands: dict[list[int]]) -> dict[list[int]]:
        """Gets the indices of the frequency band limits for each band in the
        frequencies of the results.

        PARAMETERS
        ----------
        bands : dict[list[int]]
        -   Dictionary containing the frequency bands whose results should also
            be calculated.
        -   Each key is the name of the frequency band, and each value is a list
            of numbers representing the lower- and upper-most boundaries of the
            frequency band, respectively, in the frequency units present in the
            results.

        RETURNS
        -------
        band_idcs : dict[list[int]]
        -   Indices of the frequency band limits for each band in the
            frequencies of the results.
        """

        band_freq_idcs = []
        for idx in self._results.index:
            band_idcs = deepcopy(bands)
            for name, freqs in band_idcs.items():
                band_idcs[name] = [
                    self._results["freqs"][idx].index(freq) for freq in freqs
                ]
            band_freq_idcs.append(band_idcs)

        return band_freq_idcs

    def _compute_freq_band_measure_results(
        self,
        freqs: list[Union[int, float]],
        band_values: list[list[Union[int, float]]],
        band_idcs: list[list[Union[int, float]]],
        measure: str,
    ) -> list[Union[int, float]]:
        """Computes the frequency band results for a single channel's worth of
        results.


        PARAMETERS
        ----------
        freqs : list[int | float]
        -   Frequencies of the values in the results.

        band_values : list[list[int | float]]
        -   Values of the results in each frequency band for a single channel,
            stored as a separate list.

        band_idcs : list[list[int | float]]
        -   Indices of the frequency band lower- and upper-bounds in 'freqs'.

        measure : str
        -   Measure to compute for the frequency bands.
        -   Supported inputs are: "average" for the average value; "median" for
            the median value; "max" for the maximum value; "fmax" for the
            frequency at which the maximum value occurs; "min" for the minimum
            value; and "fmin" for the frequency at which the minimum value
            occurs.

        RETURNS
        -------
        values : list[int | float]
        -   Values of the frequency band results.
        """

        if measure == "average":
            values = [float(np.mean(entry)) for entry in band_values]
        elif measure == "median":
            values = [np.median(entry) for entry in band_values]
        elif measure == "max":
            values = [np.max(entry) for entry in band_values]
        elif measure == "min":
            values = [np.min(entry) for entry in band_values]
        elif measure == "fmax" or measure == "fmin":
            if measure == "fmax":
                values = [entry.index(np.max(entry)) for entry in band_values]
            elif measure == "fmin":
                values = [entry.index(np.min(entry)) for entry in band_values]
            values = [
                list(band_idcs.values())[band_i][0] + freq_idx
                for band_i, freq_idx in enumerate(values)
            ]
            values = [freqs[freq_i] for freq_i in values]

        return values

    def _compute_freq_band_results(
        self,
        bands: dict[list[int]],
        band_freq_idcs: dict[list[int]],
        attributes: list[str],
        measures: list[str],
    ) -> None:
        """Computes the frequency band-wise results with the desired frequency
        bands on the desired result nodes.

        PARAMETERS
        ----------
        bands : dict[list[int]]
        -   Dictionary containing the frequency bands whose results should also
            be calculated.
        -   Each key is the name of the frequency band, and each value is a list
            of numbers representing the lower- and upper-most boundaries of the
            frequency band, respectively, in the frequency units present in the
            results.

        band_freq_idcs : dict[list[int]]
        -   Indices of the frequency band limits for each band in the
            frequencies of the results.

        attributes : list[str]
        -   Attributes of the results to apply the frequency band-wise analysis
            to.

        measures : list[str]
        -   Measures to compute for the frequency bands.
        -   Supported inputs are: "average" for the average value; "median" for
            the median value; "max" for the maximum value; "fmax" for the
            frequency at which the maximum value occurs; "min" for the minimum
            value; and "fmin" for the frequency at which the minimum value
            occurs.
        """

        for idx, band_idcs in enumerate(band_freq_idcs):
            for attribute in attributes:
                entries = []
                for freq_idcs in band_idcs.values():
                    entries.append(
                        self._results[attribute][idx][
                            freq_idcs[0] : freq_idcs[1] + 1
                        ]
                    )
                for measure in measures:
                    values = self._compute_freq_band_measure_results(
                        freqs=self._results["freqs"][idx],
                        band_values=entries,
                        band_idcs=band_idcs,
                        measure=measure,
                    )
                    self._results[f"{attribute}_fbands_{measure}"][idx] = values
            self._results["fband_labels"][idx] = list(bands.keys())
            self._results["fband_freqs"][idx] = list(bands.values())

    def freq_band_results(
        self, bands: dict[list[int]], attributes: list[str], measures: list[str]
    ) -> None:
        """Calculates the values of attributes in the data across specified
        frequency bands by taking the mean of these values.

        PARAMETERS
        ----------
        bands : dict[list[int]]
        -   Dictionary containing the frequency bands whose results should also
            be calculated.
        -   Each key is the name of the frequency band, and each value is a list
            of numbers representing the lower- and upper-most boundaries of the
            frequency band, respectively, in the frequency units present in the
            results.

        attributes : list[str]
        -   Attributes of the results to apply the frequency band-wise analysis
            to.

        measures : list[str]
        -   Measures to compute for the frequency bands.
        -   Supported inputs are: "average" for the average value; "median" for
            the median value; "max" for the maximum value; "freq_max" for the
            frequency at which the maximum value occurs; "min" for the minimum
            value; and "freq_min" for the frequency at which the minimum value
            occurs.

        process_idcs : list[int] | None
        -   Indices of the results to compute frequency band-wise results for.
        """

        self._prepare_fband_results(
            bands=bands, attributes=attributes, measures=measures
        )

        band_freq_idcs = self._get_band_freq_indices(bands=bands)

        self._compute_freq_band_results(
            bands=bands,
            band_freq_idcs=band_freq_idcs,
            attributes=attributes,
            measures=measures,
        )

        self._fbands = bands
        self._fband_measures = np.unique(
            [*measures, self._fband_measures]
        ).tolist()
        self._fband_desc_measures = [
            measure
            for measure in measures
            if measure in self._desc_fband_measures
        ]
        self._refresh_desc_measures()
        if self._verbose:
            print(
                "Computing frequency band-wise results for the nodes in the "
                "results with the following frequency bands (units are the "
                "same as in the results):"
            )
            print([f"{name}: {freqs}" for name, freqs in bands.items()])

    def _refresh_desc_measures(self) -> None:
        """Refreshes a list of the descriptive measures (e.g. variability
        measures such as standard error of the mean, frequency band measures
        such as the maximum values in the frequency band) that need to be
        re-calculated after any processing steps (e.g. averaging) are applied.
        """

        present_desc_measures = []
        all_measures = [*self._fband_measures, *self._var_measures]
        all_desc_measures = [
            *self._desc_fband_measures,
            *self._desc_var_measures,
        ]
        for measure in all_measures:
            if measure in all_desc_measures:
                present_desc_measures.append(measure)

        self._desc_measures = present_desc_measures

    def _prepare_var_measures(
        self,
        measures: list[str],
        process_keys: list[str],
        process_entry_idcs=list[list[int]],
    ) -> None:
        """Prepares for the calculation of variabikity measures, checking that
        the required attributes are present in the data (adding them if not)
        and checking that the requested measure is supported.

        PARAMETERS
        ----------
        measures : list[str]
        -   Types of measures to compute.
        -   Supported types are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.

        process_keys : list[str]
        -   Attributes of the results to calculate variability measures for.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if a requested variability measure is not suppoerted.
        """

        supported_measures = ["std", "sem"]
        for measure in measures:
            if measure not in supported_measures:
                raise UnavailableProcessingError(
                    "Error when calculating variability measures of the "
                    f"averaged data:\nComputing the measure '{measure}' is "
                    "not supported. Supported measures are: "
                    f"{supported_measures}"
                )

        if not self._var_measures:
            var_columns = []
            for measure in measures:
                for key in process_keys:
                    var_columns.append(f"{key}_{measure}")
            self._populate_columns(attributes=var_columns)
            self._var_columns = var_columns
        else:
            for measure in self._var_measures:
                for key in process_keys:
                    for idcs in process_entry_idcs:
                        if len(idcs) > 1:
                            self._results[f"{key}_{measure}"][idcs[0]] = None

    def _compute_var_measures(
        self,
        measures: list[str],
        process_entry_idcs: list[list[int]],
        process_keys: list[str],
    ) -> None:
        """Computes the variability measures over the unique node indices.

        PARAMETERS
        ----------
        measures : list[str]
        -   Types of variabilty measures to compute.
        -   Supported types are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.

        process_entry_indices : list[list[int]]
        -   Unique indices of nodes in the results that should be processed.

        process_keys : list[str]
        -   Attributes of the results to calculate variability measures for.
        """

        self._prepare_var_measures(
            measures=measures,
            process_keys=process_keys,
            process_entry_idcs=process_entry_idcs,
        )

        for measure in measures:
            for key in process_keys:
                results_name = f"{key}_{measure}"
                for idcs in process_entry_idcs:
                    if len(idcs) > 1:
                        entries = [self._results[key][idx] for idx in idcs]
                        if measure == "std":
                            value = np.std(entries, axis=0)
                        elif measure == "sem":
                            value = stats.sem(entries, axis=0)
                        self._results[results_name][idcs[0]] = value
        self._var_measures = list(np.unique([*self._var_measures, *measures]))

        self._refresh_desc_measures()
        if self._verbose:
            print(
                "Computing the following variability measures for the "
                f"processed data: {measures}"
            )

    def _compute_average(
        self,
        average_entry_idcs: list[list[int]],
        over_key: str,
        average_keys: list[str],
    ) -> None:
        """Computes the average results over the unique node indices.

        PARAMETERS
        ----------
        average_entry_indices : list[list[int]]
        -   Unique indices of nodes in the results that should be processed.

        over_key : str
        -   The attribute of the results to average over.

        average_keys : list[str]
        -   Attributes of the results to average.
        """

        drop_idcs = []
        for idcs in average_entry_idcs:
            if len(idcs) > 1:
                for key in average_keys:
                    entries = [self._results[key][idx] for idx in idcs]
                    self._results[key][idcs[0]] = np.mean(
                        entries, axis=0
                    ).tolist()
                drop_idcs.extend(idcs[1:])
            self._results[over_key][
                idcs[0]
            ] = f"avg{[self._results[over_key][idx] for idx in idcs]}"

        self._results = self._results.drop(index=drop_idcs)
        self._results = self._results.reset_index(drop=True)

        if self._verbose:
            print(
                f"Computing the average for {len(average_entry_idcs)} groups "
                f"of results over the '{over_key}' attribute."
            )

    def average(
        self,
        over_key: str,
        data_keys: list[str],
        group_keys: list[str],
        over_entries: Optional[list] = "ALL",
        identical_keys: Optional[list[str]] = None,
        var_measures: Optional[list[str]] = None,
    ) -> None:
        """Averages results.

        PARAMETERS
        ----------
        over_key : str
        -   Name of the attribute in the results to average over.

        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be averaged, and any variability measures computed on.

        group_keys : [list[str]]
        -   Names of the attibutes in the results to use to group results that
            will be averaged over.

        over_entries : list | "ALL"
        -   The values of the 'over_key' attribute in the results to average.
        -   If "ALL", all values of the 'over_key' attribute are included.

        identical_keys : list[str] | None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results being averaged. If they are
            not identical, an error will be raised.

        var_measures : list[str] | None
        -   Names of measures of variability to be computed alongside the
            averaging of the results.
        -   Supported measures are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.
        """

        if over_entries == "ALL":
            over_entries = list(np.unique(self._results[over_key]))

        combined_entries = self._get_combined_entries(group_keys=group_keys)
        unique_entry_idcs = self._get_unique_indices(
            combined_entries=combined_entries
        )

        if identical_keys is not None:
            self._check_identical_keys(
                indices=unique_entry_idcs, keys=identical_keys
            )

        average_entry_idcs = self._get_indices_to_process(
            unique_entry_idcs=unique_entry_idcs,
            over_key=over_key,
            over_entries=over_entries,
        )

        average_keys = self._get_process_keys(
            process_keys=data_keys,
        )

        if var_measures:
            self._compute_var_measures(
                measures=var_measures,
                process_entry_idcs=average_entry_idcs,
                process_keys=average_keys,
            )

        self._compute_average(
            average_entry_idcs=average_entry_idcs,
            over_key=over_key,
            average_keys=average_keys,
        )

        if self._fbands:
            self.freq_band_results(
                bands=self._fbands,
                attributes=data_keys,
                measures=self._fband_desc_measures,
            )

        print("jeff")


def load_results_of_types(
    results_folderpath: str,
    to_analyse: dict[str],
    result_types: list[str],
    extract_from_dicts: Optional[dict[list[str]]] = None,
    identical_entries: Optional[list[str]] = None,
    discard_entries: Optional[list[str]] = None,
    allow_missing: bool = False,
) -> PostProcess:
    """Loads results of a multiple types and merges them into a single
    PostProcess object.

    PARAMETERS
    ----------
    results_folderpath : str
    -   Folderpath to where the results are located.

    to_analyse : dict[str]
    -   Dictionary in which each entry represents a different piece of results.
    -   Contains the keys: 'sub' (subject ID); 'ses' (session name); 'task'
        (task name); 'acq' (acquisition type); and 'run' (run number).

    result_types : list[str]
    -   The types of results to analyse.

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_entries : list[str] | None; default None
    -   The entries in 'results' which are identical across channels and for
        which only one copy is present.

    discard_entries : list[str] | None; default None
    -   The entries which should be discarded immediately without
        processing.

    allow_missing : bool; default False
    -   Whether or not to allow new rows to be present in the merged results
        with NaN values for columns not shared between the results being
        merged if the shared columns do not have matching values.
    -   I.e. if you want to make sure you are merging results from the same
        channels, set this to False, otherwise results from different
        channels will be merged and any missing information will be set to
        NaN.

    RETURNS
    -------
    all_results : PostProcess
    -   The results merged across the specified result types.
    """

    first_type = True
    for result_type in result_types:
        results = load_results_of_type(
            results_folderpath=results_folderpath,
            to_analyse=to_analyse,
            result_type=result_type,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
        )
        if first_type:
            all_results = deepcopy(results)
            first_type = False
        else:
            all_results.merge_from_df(
                new_results=deepcopy(results.as_df()),
                allow_missing=allow_missing,
            )

    return all_results


def load_results_of_type(
    results_folderpath: str,
    to_analyse: list[dict[str]],
    result_type: str,
    extract_from_dicts: Optional[dict[list[str]]] = None,
    identical_entries: Optional[list[str]] = None,
    discard_entries: Optional[list[str]] = None,
) -> PostProcess:
    """Loads results of a single type and appends them into a single PostProcess
    object.

    PARAMETERS
    ----------
    results_folderpath : str
    -   Folderpath to where the results are located.

    to_analyse : list[dict[str]]
    -   Dictionary in which each entry represents a different piece of results.
    -   Contains the keys: 'sub' (subject ID); 'ses' (session name); 'task'
        (task name); 'acq' (acquisition type); and 'run' (run number).

    result_type : str
    -   The type of results to analyse.

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_entries : list[str] | None; default None
    -   The entries in 'results' which are identical across channels and for
        which only one copy is present.

    discard_entries : list[str] | None; default None
    -   The entries which should be discarded immediately without
        processing.

    RETURNS
    -------
    results : PostProcess
    -   The appended results for a single type of data.
    """

    first_result = True
    for result_info in to_analyse:
        result_fpath = generate_results_fpath(
            folderpath=results_folderpath,
            subject=result_info["sub"],
            session=result_info["ses"],
            task=result_info["task"],
            acquisition=result_info["acq"],
            run=result_info["run"],
            result_type=result_type,
            filetype=".json",
        )
        result = load_file(fpath=result_fpath)
        if first_result:
            results = PostProcess(
                results=result,
                extract_from_dicts=extract_from_dicts,
                identical_entries=identical_entries,
                discard_entries=discard_entries,
            )
            first_result = False
        else:
            results.append_from_dict(
                new_results=result,
                extract_from_dicts=extract_from_dicts,
                identical_entries=identical_entries,
                discard_entries=discard_entries,
            )

    return results
