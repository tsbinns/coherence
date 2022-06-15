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
from scipy import stats
from coh_handle_files import generate_results_fpath, load_file
from coh_exceptions import (
    DuplicateEntryError,
    EntryLengthError,
    ProcessingOrderError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_entries import (
    combine_col_vals_df,
    check_non_repeated_vals_lists,
    check_vals_identical_df,
    dict_to_df,
    get_eligible_idcs_list,
    get_group_idcs,
    sort_inputs_results,
)
from coh_saving import save_dict, save_object


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
    -   Keys ending with "_dimensions" are treated as containing information
        about the dimensions of other attributes in the results, e.g.
        'X_dimensions' would specify the dimensions for attribute 'X'. The
        dimensions should be a list of strings containing the values "channels"
        and "frequencies" in the positions corresponding to the axis of these
        dimensions in 'X'. A single list should be given, i.e. 'X_dimensions'
        should hold for all entries of 'X'. If no dimensions, are given, the 0th
        axis is assumed to correspond to channels and the 1st axis to
        frequencies.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        ['epochs', 'channels', 'frequencies', 'timepoints'], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions ["channels", "frequencies", "epochs", "timepoints"].
    -   If the dimensions is a list of lists of strings, there should be a
        sublist for each channel/node in the results, and hence no "channel"
        axis should be present in the dimensions of an individual node/channel.

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.
    -   If any dimension attributes are present, these should be included as an
        identical entry, as they will be added automatically.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without processing.

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
        identical_keys: Optional[list[str]] = None,
        discard_keys: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> None:

        # Initialises inputs of the object.
        results = sort_inputs_results(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=verbose,
        )
        self._results = dict_to_df(obj=results)
        self._verbose = verbose

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._process_measures = []
        self._fbands = None
        self._fband_attributes = None
        self._fband_measures = []
        self._fband_desc_measures = []
        self._fband_columns = []
        self._var_measures = []
        self._var_columns = []
        self._desc_measures = ["n_from"]
        self._desc_process_measures = ["n_from"]
        self._desc_fband_measures = ["max", "min", "fmax", "fmin"]
        self._desc_var_measures = ["std", "sem"]

    def append_from_dict(
        self,
        new_results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_keys: Optional[list[str]] = None,
        discard_keys: Optional[list[str]] = None,
    ) -> None:
        """Appends a dictionary of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

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

        identical_keys : list[str] | None; default None
        -   The keys in 'results' which are identical across channels and for
            which only one copy is present.

        discard_keys : list[str] | None; default None
        -   The keys which should be discarded immediately without
            processing.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if frequency band results have been added.
        """

        if self._fbands:
            raise ProcessingOrderError(
                "Error when trying to add results:\nNew results cannot be "
                "added after frequency band-wise results have been calculated."
            )

        new_results = sort_inputs_results(
            results=new_results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=self._verbose,
        )

        check_non_repeated_vals_lists(
            lists=[list(self._results.keys()), list(new_results.keys())],
            allow_non_repeated=False,
        )

        new_results = dict_to_df(obj=new_results)

        self._results = pd.concat(
            objs=[self._results, new_results], ignore_index=True
        )

    def append_from_df(
        self,
        new_results: pd.DataFrame,
    ) -> None:
        """Appends a DataFrame of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   The new results to append.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if frequency band results have been added.
        """

        if self._fbands:
            raise ProcessingOrderError(
                "Error when trying to add results:\nNew results cannot be "
                "added after frequency band-wise results have been calculated."
            )

        check_non_repeated_vals_lists(
            lists=[self._results.keys().tolist(), new_results.keys().tolist()],
            allow_non_repeated=False,
        )

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
                        dataframe.at[row_i, key] = tuple(dataframe[key][row_i])
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
                    results.at[row_i, key] = list(results[key][row_i])

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

        all_repeated = check_non_repeated_vals_lists(
            lists=[self._results.keys().tolist(), new_results.keys().tolist()],
            allow_non_repeated=True,
        )

        if all_repeated:
            raise DuplicateEntryError(
                "Error when trying to merge results:\nThere are no new columns "
                "in the results being added. If you still want to add the "
                "results, use the append methods."
            )

    def merge_from_dict(
        self,
        new_results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_keys: Optional[list[str]] = None,
        discard_keys: Optional[list[str]] = None,
        allow_missing: bool = False,
    ) -> None:
        """Merges a dictionary of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

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

        identical_keys : list[str] | None; default None
        -   The keys in 'results' which are identical across channels and for
            which only one copy is present.

        discard_keys : list[str] | None; default None
        -   The keys which should be discarded immediately without
            processing.

        allow_missing : bool; default False
        -   Whether or not to allow new rows to be present in the merged results
            with NaN values for columns not shared between the results being
            merged if the shared columns do not have matching values.
        -   I.e. if you want to make sure you are merging results from the same
            channels, set this to False, otherwise results from different
            channels will be merged and any missing information will be set to
            NaN.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if frequency band results have been added.
        """

        if self._fbands:
            raise ProcessingOrderError(
                "Error when trying to add results:\nNew results cannot be "
                "added after frequency band-wise results have been calculated."
            )

        new_results = sort_inputs_results(
            results=new_results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=self._verbose,
        )

        new_results = dict_to_df(obj=new_results)

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
        -   Cannot be called after frequency band results have been computed.

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

        RAISES
        ------
        ProcessingOrderError
        -   Raised if frequency band results have been added.
        """

        if self._fbands:
            raise ProcessingOrderError(
                "Error when trying to add results:\nNew results cannot be "
                "added after frequency band-wise results have been calculated."
            )

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
                if key in self._fband_attributes:
                    fband_keys.extend(
                        [
                            f"{key}_fbands_{measure}"
                            for measure in self._fband_measures
                            if measure not in self._desc_fband_measures
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
                for freq in freqs:
                    if freq not in self._results["freqs"][idx]:
                        raise ValueError(
                            "Error when trying to compute the frequency "
                            "band-wise results:\nThe frequencies in the range "
                            f"{freqs[0]} - {freqs[1]} (units identical to "
                            "those in the results) are not present in the "
                            "results with frequency range "
                            f"{self._results['freqs'][0]} - "
                            f"{self._results['freqs'][1]}."
                        )
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
                    self._results.at[
                        idx, f"{attribute}_fbands_{measure}"
                    ] = values
            self._results.at[idx, "fband_labels"] = list(bands.keys())
            self._results.at[idx, "fband_freqs"] = list(bands.values())

    def freq_band_results(
        self, bands: dict[list[int]], attributes: list[str], measures: list[str]
    ) -> None:
        """Calculates the values of attributes in the data across specified
        frequency bands by taking the mean of these values.
        -   Once called, these frequency band values, attributes, and measures
            will be used in all further processing (e.g. averaging), and this
            method cannot be called again.

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

        RAISES
        ------
        ProcessingOrderError
        -   Raised if frequency band results have already been computed.
        """

        if self._fbands:
            raise ProcessingOrderError(
                "Error when trying to compute the frequency band results:\n"
                "Frequency band results have already been computed. If you "
                "want to compute results with different settings, you must use "
                "a fresh PostProcess object."
            )
        if self._verbose:
            print(
                "Computing frequency band-wise results for the nodes in the "
                "results with the following frequency bands (units are the "
                "same as in the results):"
            )
            print([f"{name}: {freqs}" for name, freqs in bands.items()])

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
        self._fband_attributes = attributes
        self._fband_measures = measures
        self._fband_desc_measures = [
            measure
            for measure in measures
            if measure in self._desc_fband_measures
        ]

    def _refresh_freq_band_results(self) -> None:
        """Recomputes the value of frequency band result descriptive measures
        (i.e. maximum, minimum, and their frequencies) following processing
        (e.g. averaging).
        """

        self._prepare_fband_results(
            bands=self._fbands,
            attributes=self._fband_attributes,
            measures=self._fband_desc_measures,
        )

        band_freq_idcs = self._get_band_freq_indices(bands=self._fbands)

        self._compute_freq_band_results(
            bands=self._fbands,
            band_freq_idcs=band_freq_idcs,
            attributes=self._fband_attributes,
            measures=self._fband_desc_measures,
        )

    def _refresh_desc_measures(self) -> None:
        """Refreshes a list of the descriptive measures (e.g. variability
        measures such as standard error of the mean, frequency band measures
        such as the maximum values in the frequency band) that need to be
        re-calculated after any processing steps (e.g. averaging) are applied.
        """

        present_desc_measures = []
        all_measures = [
            *self._process_measures,
            *self._fband_measures,
            *self._var_measures,
        ]
        all_desc_measures = [
            *self._desc_process_measures,
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

        current_measures = np.unique([*self._var_measures, *measures]).tolist()
        for measure in current_measures:
            for key in process_keys:
                attribute_name = f"{key}_{measure}"
                if attribute_name not in self._results.keys():
                    self._populate_columns(attributes=[attribute_name])
                else:
                    for idcs in process_entry_idcs:
                        self._results.at[idcs[0], attribute_name] = None

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
                            value = np.std(entries, axis=0).tolist()
                        elif measure == "sem":
                            value = stats.sem(entries, axis=0).tolist()
                        self._results.at[idcs[0], results_name] = value
        self._var_measures = np.unique(
            [*self._var_measures, *measures]
        ).tolist()

        self._refresh_desc_measures()
        if self._verbose:
            print(
                "Computing the following variability measures on attributes "
                "for the processed data:\n- Variability measure(s): "
                f"{measures}\n- On attribute(s): {process_keys}\n"
            )

    def _set_averaged_key_value(
        self, key: str, idcs: list[int], if_one_value: bool = True
    ) -> None:
        """Sets the value for attributes in the nodes of the results being
        averaged together based on the unique values of the attributes at these
        nodes.
        -   E.g. averaging over two nodes of an attribute with the values '1'
            and '2', respectively, would be transformed into: 'avg[1, 2]'.
        -   Equally, averaging over three nodes of an attribute with the values
            '1', '1', and '2' would be transformed into: 'avg[1, 2]', as only
            the unique values are accounted for.

        PARAMETERS
        ----------
        key : str
        -   Name of the attribute in the data.

        idcs : list[int]
        -   Indices of nodes in the results being averaged together.

        if_one_value : bool; default True
        -   Whether or not to change the key value if the attribute of the nodes
            being averaged together contains only a single unique value.
        -   If True and only one unique value is present, the value will be
            transformed, otherwise if False, the value will not be transformed.
        """

        if if_one_value:
            min_length = 0
        else:
            min_length = 2

        entries = np.unique([self._results[key][idx] for idx in idcs]).tolist()
        if len(entries) >= min_length:
            value = "avg["
            for entry in entries:
                value += f"{str(entry)}, "
            value = value[:-2] + "]"
            self._results.at[idcs[0], key] = value

    def _compute_average(
        self,
        average_entry_idcs: list[list[int]],
        over_key: str,
        average_keys: list[str],
        group_keys: list[str],
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

        group_keys : list[str]
        -   Attributes of the results whose entries should be changed to reflect
            which entries have been averaged over.
        """

        drop_idcs = []
        for idcs in average_entry_idcs:
            if len(idcs) > 1:
                for key in average_keys:
                    entries = [self._results[key][idx] for idx in idcs]
                    self._results.at[idcs[0], key] = np.mean(
                        entries, axis=0
                    ).tolist()
                drop_idcs.extend(idcs[1:])
                for key in group_keys:
                    self._set_averaged_key_value(
                        key=key, idcs=idcs, if_one_value=False
                    )
            self._set_averaged_key_value(
                key=over_key, idcs=idcs, if_one_value=True
            )
            self._results.at[idcs[0], "n_from"] = len(idcs)

        self._results = self._results.drop(index=drop_idcs)
        self._results = self._results.reset_index(drop=True)

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

        if self._verbose:
            print(
                "Computing the average for groups of results:\nAverage-over "
                f"attribute: {over_key}\nAverage-over attribute value(s): "
                f"{over_entries}\nData attribute(s): {data_keys}\nGrouping "
                f"attribute(s): {group_keys}\nCheck identical across results "
                f"attribute(s): {identical_keys}\nVariability measure(s): "
                f"{var_measures}\n"
            )

        eligible_idcs = get_eligible_idcs_list(
            vals=self._results[over_key],
            eligible_vals=over_entries,
        )
        combined_vals = combine_col_vals_df(
            dataframe=self._results,
            keys=group_keys,
            idcs=eligible_idcs,
            special_vals={"avg[": "avg_"},
        )
        group_idcs, _ = get_group_idcs(
            vals=combined_vals, replacement_idcs=eligible_idcs
        )

        if identical_keys is not None:
            check_vals_identical_df(
                dataframe=self._results,
                keys=identical_keys,
                idcs=group_idcs,
            )
        average_keys = self._get_process_keys(
            process_keys=data_keys,
        )

        if var_measures:
            self._compute_var_measures(
                measures=var_measures,
                process_entry_idcs=group_idcs,
                process_keys=average_keys,
            )

        self._compute_average(
            average_entry_idcs=group_idcs,
            over_key=over_key,
            average_keys=average_keys,
            group_keys=group_keys,
        )

        if self._fbands:
            self._refresh_freq_band_results()

    def results_as_df(self) -> pd.DataFrame:
        """Returns the results as a pandas DataFrame.

        RETURNS
        -------
        results : pandas DataFrame
        -   The results as a pandas DataFrame.
        """

        return self._results

    def _check_attrs_identical(self, attrs: list[str]) -> None:
        """Checks that the values of attributes in the results are identical
        across nodes.

        PARAMETERS
        ----------
        attrs : list[str]
        -   Names of the attributes to check in the results.

        RAISES
        ------
        UnidenticalEntryError
        -   Raised if the values of an attribute are not identical across the
            nodes in the results.
        """

        for attr in attrs:
            attr_vals = self._results[attr].tolist()
            if len(np.unique(attr_vals)) != 1:
                raise UnidenticalEntryError(
                    "Error when checking whether values belonging to an "
                    "attribute of the results are identical:\nValues of the "
                    f"attribute '{attr}' are not identical."
                )

    def _sequester_to_dicts(
        self, sequester: dict[list[str]]
    ) -> tuple[dict[dict], list[str]]:
        """Sequesters attributes of the results into dictionaries within a
        parent dictionary

        PARAMETERS
        ----------
        sequester : dict[list[str]]
        -   Attributes of the results to sequester into dictionaries within the
            returned results dictionary.
        -   Each key is the name of the dictionary that the attributes (the
            values for this key given as strings within a list corresponding to
            the names of the attributes) will be included in.
        -   E.g. an 'extract_to_dicts' of {"metadata": ["subject", "session"]}
            would create a dictionary within the returned dictionary of results
            of {"metadata": {"subject": VALUE, "session": VALUE}}.
        -   Values of extracted attributes must be identical for each node in
            the results, so that the dictionary the value is sequestered into
            contains only a single value for each attribute.

        RETURNS
        -------
        results : dict[dict]
        -   Dictionary with the requested attributes sequestered into the
            requested dictionaries.

        attrs_to_sequester : list[str]
        -   Names of attributes in the results that have been sequestered into
            dictionaries.
        """

        attrs_to_sequester = []
        for values in sequester.values():
            attrs_to_sequester.extend(values)
        self._check_attrs_identical(attrs=attrs_to_sequester)

        results = {}
        for dict_name, attrs in sequester.items():
            results[dict_name] = {}
            for attr in attrs:
                results[dict_name][attr] = self._results[attr][0]

        return results, attrs_to_sequester

    def results_as_dict(
        self, sequester_to_dicts: Union[dict[list[str]], None] = None
    ) -> dict:
        """Converts the results from a pandas DataFrame to a dictionary and
        returns the results.

        PARAMETERS
        ----------
        sequester_to_dicts : dict[list[str]] | None; default None
        -   Attributes of the results to sequester into dictionaries within the
            returned results dictionary.
        -   Each key is the name of the dictionary that the attributes (the
            values for this key given as strings within a list corresponding to
            the names of the attributes) will be included in.
        -   E.g. an 'extract_to_dicts' of {"metadata": ["subject", "session"]}
            would create a dictionary within the returned dictionary of results
            of {"metadata": {"subject": VALUE, "session": VALUE}}.
        -   Values of extracted attributes must be identical for each node in
            the results, so that the dictionary the value is sequestered into
            contains only a single value for each attribute.

        RETURNS
        -------
        results : dict
        -   The results as a dictionary.
        """

        if sequester_to_dicts is not None:
            results, ignore_attrs = self._sequester_to_dicts(
                sequester=sequester_to_dicts
            )
        else:
            results = {}
            ignore_attrs = []

        for attr in self._results.keys():
            if attr not in ignore_attrs:
                results[attr] = self._results[attr].copy().tolist()

        return results

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the PostProcess object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   Whether or not the user is asked to confirm to overwrite a
            pre-existing file if one exists.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_object(
            to_save=self,
            fpath=fpath,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        sequester_to_dicts: Union[dict[list[str]], None] = None,
        ask_before_overwrite: Union[bool, None] = None,
    ) -> None:
        """Saves the results and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved.

        ftype : str | None; default None
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        sequester_to_dicts : dict[list[str]] | None; default None
        -   Attributes of the results to sequester into dictionaries within the
            returned results dictionary.
        -   Each key is the name of the dictionary that the attributes (the
            values for this key given as strings within a list corresponding to
            the names of the attributes) will be included in.
        -   E.g. an 'extract_to_dicts' of {"metadata": ["subject", "session"]}
            would create a dictionary within the returned dictionary of results
            of {"metadata": {"subject": VALUE, "session": VALUE}}.
        -   Values of extracted attributes must be identical for each node in
            the results, so that the dictionary the value is sequestered into
            contains only a single value for each attribute.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_dict(
            to_save=self.results_as_dict(sequester_to_dicts=sequester_to_dicts),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )


def load_results_of_types(
    folderpath_processing: str,
    to_analyse: dict[str],
    result_types: list[str],
    extract_from_dicts: Optional[dict[list[str]]] = None,
    identical_keys: Optional[list[str]] = None,
    discard_keys: Optional[list[str]] = None,
    allow_missing: bool = False,
) -> PostProcess:
    """Loads results of a multiple types and merges them into a single
    PostProcess object.

    PARAMETERS
    ----------
    folderpath_processing : str
    -   Folderpath to where the processed results are located.

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

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without
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
            folderpath_processing=folderpath_processing,
            to_analyse=to_analyse,
            result_type=result_type,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
        )
        if first_type:
            all_results = deepcopy(results)
            first_type = False
        else:
            all_results.merge_from_df(
                new_results=deepcopy(results.results_as_df()),
                allow_missing=allow_missing,
            )

    return all_results


def load_results_of_type(
    folderpath_processing: str,
    to_analyse: list[dict[str]],
    result_type: str,
    extract_from_dicts: Optional[dict[list[str]]] = None,
    identical_keys: Optional[list[str]] = None,
    discard_keys: Optional[list[str]] = None,
) -> PostProcess:
    """Loads results of a single type and appends them into a single PostProcess
    object.

    PARAMETERS
    ----------
    folderpath_processing : str
    -   Folderpath to where the processed results are located.

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

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without
        processing.

    RETURNS
    -------
    results : PostProcess
    -   The appended results for a single type of data.
    """

    first_result = True
    for result_info in to_analyse:
        result_fpath = generate_results_fpath(
            folderpath=folderpath_processing,
            dataset=result_info["cohort"],
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
                identical_keys=identical_keys,
                discard_keys=discard_keys,
            )
            first_result = False
        else:
            results.append_from_dict(
                new_results=result,
                extract_from_dicts=extract_from_dicts,
                identical_keys=identical_keys,
                discard_keys=discard_keys,
            )

    return results
