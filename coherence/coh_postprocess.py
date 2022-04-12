"""Class for applying post-processing to data.

CLASSES
-------
PostProcess
-   Class for the post-processing of results derived from raw signals.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Union
from numpy.typing import NDArray

from coh_exceptions import (
    EntryLengthError,
    InputTypeError,
    MissingEntryError,
    PreexistingAttributeError,
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
    ) -> None:

        # Initialises inputs of the object.
        results = self._sort_inputs(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
        )

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._results = self._results_to_df(results=results)

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
            del results[entry]

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
        processing.

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
        pandas DataFrame
        -   The 'results' dictionary as a DataFrame.
        """

        return pd.DataFrame.from_dict(data=results, orient="columns")

    def _check_non_duplicates(self, results: dict) -> None:
        """Checks that the existing results and the results being added contain
        all the same keys.

        PARAMETERS
        ----------
        results : dict
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

    def append(
        self,
        results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_entries: Optional[list[str]] = None,
        discard_entries: Optional[list[str]] = None,
    ) -> None:
        """Appends other dictionaries of results to the list of result
        dictionaries stored in the PostProcess object."""

        results = self._sort_inputs(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
        )

        self._check_non_duplicates(results=results)

        results = self._results_to_df(results=results)

        self._results = pd.concat(
            objs=[self._results, results], ignore_index=True
        )
