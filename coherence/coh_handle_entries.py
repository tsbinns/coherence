"""Contains functions and classes for handling entries within objects.

CLASSES
-------
FillerObject
-   Empty object that can be filled with attributes that would otherwise not be
    accessible as a class attribute.

METHODS
-------
check_lengths_dict_identical
-   Checks whether the lengths of entries within a dictionary are identical.

check_lengths_dict_equals_n
-   Checks whether the lengths of entries within a dictionary is equal to a
    given number.

check_lengths_list_identical
-   Checks whether the lengths of entries within a list are identical.

check_lengths_list_equals_n
-   Checks whether the lengths of entries within a list is equal to a given
    number.

check_repeated_vals
-   Checks whether duplicates exist within an input list.

check_matching_entries
-   Checks whether the entries of objects match one another.

check_master_entries_in_sublists
-   Checks whether all values in a master list are present in a set of sublists.

check_sublist_entries_in_master
-   Checks whether all values in a set of sublists are present in a master list.

ordered_list_from_dict
-   Creates a list from entries in a dictionary, sorted based on a given order.

ordered_dict_from_list
-   Creates a dictionary with keys occuring in a given order.

ragged_array_to_list
-   Converts a ragged numpy array of nested arrays to a ragged list of nested
    lists.

drop_from_list
-   Drops specified entries from a list.

drop_from_dict
-   Removes specified entries from a dictionary.

sort_inputs_results
-   Checks that the values in 'results' are in the appropriate format for
    processing with PostProcess or Plotting class objects.

dict_to_df
-   Converts a dictionary into a pandas DataFrame.
"""

from copy import deepcopy
from itertools import chain
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from coh_exceptions import (
    DuplicateEntryError,
    EntryLengthError,
    MissingEntryError,
    PreexistingAttributeError,
    UnidenticalEntryError,
)


class FillerObject:
    """Creates an empty object that can be filled with attributes that would
    otherwise not be accessible as a class attribute."""


def _find_lengths_dict(
    to_check: dict,
    ignore_values: Optional[list],
    ignore_keys: Optional[list],
) -> list[int]:
    """Finds the lengths of entries within a dictionary.

    PARAMETERS
    ----------
    to_check : dict
    -   The dictionary for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    ignore_keys : list | None; default None
    -   The keys of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no keys are ignored.

    RETURNS
    -------
    entry_lengths : list[int]
    -   The lengths of entries in the list.
    """

    entry_lengths = []
    for key, value in to_check.items():
        if key not in ignore_keys or value not in ignore_values:
            entry_lengths.append(len(value))

    return entry_lengths


def check_lengths_dict_identical(
    to_check: dict,
    ignore_values: Optional[list] = None,
    ignore_keys: Optional[list] = None,
) -> tuple[bool, Union[int, list[int]]]:
    """Checks whether the lengths of entries in the input dictionary are
    identical.

    PARAMETERS
    ----------
    to_check : dict
    -   The dictionary for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    ignore_keys : list | None; default None
    -   The keys of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no keys are ignored.

    RETURNS
    -------
    identical : bool
    -   Whether or not the lengths of the entries are identical.

    lengths : int | list
    -   The length(s) of the entries. If the lengths are identical,
        'lengths' is an int representing the length of all items.
    -   If the lengths are not identical, 'lengths' is a list containing the
        lengths of the individual entries (i.e. 'entry_lengths').
    """

    entry_lengths = _find_lengths_dict(
        to_check=to_check, ignore_values=ignore_values, ignore_keys=ignore_keys
    )

    if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
        identical = True
        lengths = entry_lengths[0]
    else:
        identical = False
        lengths = entry_lengths

    return identical, lengths


def check_lengths_dict_equals_n(
    to_check: dict,
    n: int,
    ignore_values: Optional[list] = None,
    ignore_keys: Optional[list] = None,
) -> bool:
    """Checks whether the lengths of entries in the input dictionary are equal
    to a given number.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    n : int
    -   The integer which the lengths of the entries should be equal to.

    RETURNS
    -------
    all_n : bool
    -   Whether or not the lengths of the entries are equal to 'n'.
    """

    entry_lengths = _find_lengths_dict(
        to_check=to_check, ignore_values=ignore_values, ignore_keys=ignore_keys
    )

    if entry_lengths.count(n) == len(entry_lengths):
        all_n = True
    else:
        all_n = False

    return all_n


def _find_lengths_list(
    to_check: list, ignore_values: Optional[list], axis: int
) -> list[int]:
    """Finds the lengths of entries within a list.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None, no values are ignored.

    axis : int
    -   The axis of the list whose lengths should be checked.

    RETURNS
    -------
    entry_lengths : list[int]
    -   The lengths of entries in the list.
    """

    if ignore_values is None:
        ignore_values = []
    entry_lengths = []
    for value in to_check:
        if value not in ignore_values:
            value = np.asarray(value, dtype=object)
            entry_lengths.append(np.shape(value)[axis])

    return entry_lengths


def check_lengths_list_identical(
    to_check: list, ignore_values: Optional[list] = None, axis: int = 0
) -> tuple[bool, Union[int, list[int]]]:
    """Checks whether the lengths of entries in the input list are identical.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    axis : int | default 0
    -   The axis of the list whose length should be checked.

    RETURNS
    -------
    identical : bool
    -   Whether or not the lengths of the entries are identical.

    lengths : int | list
    -   The length(s) of the entries. If the lengths are identical,
        'lengths' is an int representing the length of all items.
    -   If the lengths are not identical, 'lengths' is a list containing the
        lengths of the individual entries (i.e. 'entry_lengths').
    """

    entry_lengths = _find_lengths_list(
        to_check=to_check, ignore_values=ignore_values, axis=axis
    )

    if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
        identical = True
        lengths = entry_lengths[0]
    else:
        identical = False
        lengths = entry_lengths

    return identical, lengths


def check_lengths_list_equals_n(
    to_check: list, n: int, ignore_values: Optional[list] = None, axis: int = 0
) -> bool:
    """Checks whether the lengths of entries in the input dictionary are equal
    to a given number.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    n : int
        -   The integer which the lengths of the entries should be equal to.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    axis : int | default 0
    -   The axis of the list whose lengths should be checked.

    RETURNS
    -------
    all_n : bool
    -   Whether or not the lengths of the entries are equal to 'n'.
    """

    entry_lengths = _find_lengths_list(
        to_check=to_check, ignore_values=ignore_values, axis=axis
    )

    if entry_lengths.count(n) == len(entry_lengths):
        all_n = True
    else:
        all_n = False

    return all_n


def check_vals_identical(to_check: list) -> tuple[bool, Union[list, None]]:
    """Checks whether all values within a list are identical.

    PARAMETERS
    ----------
    to_check : list
    -   The list whose values should be checked.

    RETURNS
    -------
    is_identical : bool
    -   Whether or not all values within the list are identical.

    unique_vals : list | None
    -   The unique values in the list. If all values are identical, this is
        'None'.
    """

    is_identical = True
    compare_against = to_check[0]
    for val in to_check[1:]:
        if val != compare_against:
            is_identical = False

    if is_identical:
        unique_vals = None
    else:
        unique_vals = np.unique(to_check).tolist()

    return is_identical, unique_vals


def check_vals_identical_df(
    dataframe: pd.DataFrame, keys: list[str], idcs: list[list[int]]
) -> None:
    """Checks that a DataFrame attribute's values at specific indices are
    identical.

    PARAMETERS
    ----------
    dataframe : pandas DataFrame
    -   DataFrame containing the values to check.

    keys : list[str]
    -   Names of the attributes in the DataFrame whose values should be checked.

    idcs : list[list[int]]
    -   The indices of the entries in the attributes whose values should be
        checked.
    -   Each entry is a list of integers corresponding to the indices of
        the results to compare together.

    RAISES
    ------
    UnidenticalEntryError
    -   Raised if any of the groups of values being compared are not identical.
    """

    for key in keys:
        for group_idcs in idcs:
            if len(group_idcs) > 1:
                is_identical, unique_vals = check_vals_identical(
                    to_check=dataframe[key].iloc[group_idcs].tolist()
                )
                if not is_identical:
                    raise UnidenticalEntryError(
                        "Error when checking that the attributes of "
                        "results belonging to the same group share the "
                        f"same values:\nThe values of '{key}' in rows "
                        f"{group_idcs} do not match.\nValues:{unique_vals}\n"
                    )


def get_eligible_idcs_list(
    vals: list,
    eligible_vals: list,
    idcs: Union[list[int], None] = None,
) -> list[int]:
    """Finds indices of items in a list that have a certain value.

    PARAMETERS
    ----------
    vals : list
    -   List whose values should be checked.

    eligible_vals : list
    -   List containing values that are considered 'eligible', and whose indices
        will be recorded.

    idcs : list[int] | None; default None
    -   Indices of the items in 'to_check' to check.
    -   If 'None', all items are checked.

    RETURNS
    -------
    list[int]
    -   List containing the indices of items in 'to_check' with 'eligible'
        values.
    """

    if idcs is None:
        idcs = range(len(vals))

    return [idx for idx in idcs if vals[idx] in eligible_vals]


def get_group_idcs(
    vals: list, replacement_idcs: Union[list[int], None] = None
) -> list[list[int]]:
    """Finds groups of items in a list containing the same values, and returns
    their indices.

    PARAMETERS
    ----------
    vals : list
    -   List containing the items that should be compared.

    replacement_idcs : list[int] | None
    -   List containing indices that the indices of items in 'vals' should be
        replaced with.
    -   Must have the same length as 'vals'.
    -   E.g. if items in positions 0, 1, and 2 of 'vals' were grouped together
        and the values of 'replacement_idcs' in positions 0 to 2 were [2, 6, 9],
        respectively, the resulting indices for this group would be [2, 6, 9].
    -   If None, the original indices are used.

    RETURNS
    -------
    group_idcs : list[list[int]]
    -   List of lists where each list contains the indices for a group of items
        in 'vals' that share the same value.

    RAISES
    ------
    EntryLengthError
    -   Raised if 'vals' and 'replacement_idcs' do not have the same length.
    """

    if replacement_idcs is None:
        replacement_idcs = range(len(vals))
    else:
        if len(replacement_idcs) != len(vals):
            raise EntryLengthError(
                "Error when trying to find the group indices of items:\nThe "
                "values and replacement indices do not have the same lengths "
                f"({len(vals)} and {len(replacement_idcs)}, respectively).\n"
            )

    unique_vals = np.unique(vals).tolist()
    group_idcs = []
    for unique_val in unique_vals:
        group_idcs.append([])
        for idx, val in enumerate(vals):
            if unique_val == val:
                group_idcs[-1].append(replacement_idcs[idx])

    return group_idcs


def combine_col_vals_df(
    dataframe: pd.DataFrame,
    keys: Union[list[str], None] = None,
    idcs: Union[list[int], None] = None,
    special_vals: Union[dict[str], None] = None,
) -> list[str]:
    """Combines the values of DataFrame columns into a string on a row-by-row
    basis (i.e. one string for each row).

    PARAMETERS
    ----------
    dataframe : pandas DataFrame
    -   DataFrame whose values should be combined across columns.

    keys : list[str] | None
    -   Names of the columns in the DataFrame whose values should be combined.
    -   If 'None', all columns are used.

    idcs : list[int] | None
    -   Indices of the rows in the DataFrame whose values should be combined.
    -   If 'None', all rows are used.

    special_vals : dict[str] | None
    -   Instructions for how to treat specific values in the DataFrame.
    -   Keys are the special values that the values should begin with, whilst
        values are the values that the special values should be replaced with.
    -   E.g. {"avg[": "avg_"} would mean values in the DataFrame beginning with
        'avg[' would have this beginning replaced with 'avg_', followed by the
        column name, so a value beginning with 'avg[' in the 'channels' column
        would become 'avg_channels'.

    RETURNS
    -------
    combined_vals : list[str]
    -   The values of the DataFrame columns combined on a row-by-row basis, with
        length equal to that of 'idcs'.
    """

    if keys is None:
        keys = dataframe.keys().tolist()
    if idcs is None:
        idcs = dataframe.index.tolist()
    if special_vals is None:
        special_vals = {}

    combined_vals = []
    for idx in idcs:
        combined_vals.append("")
        for key in keys:
            value = str(dataframe[key].iloc[idx])
            for to_replace, replacement in special_vals.items():
                if value[: len(to_replace)] == to_replace:
                    value = f"{replacement}{key}"
            combined_vals[idx] += value

    return combined_vals


def check_repeated_vals(
    to_check: list,
) -> tuple[bool, Optional[list]]:
    """Checks whether repeated values exist within an input list.

    PARAMETERS
    ----------
    to_check : list
    -   The list of values whose entries should be checked for repeats.

    RETURNS
    -------
    repeats : bool
    -   Whether or not repeats are present.

    repeated_vals : list | None
    -   The list of repeated values, or 'None' if no repeats are present.
    """

    seen = set()
    seen_add = seen.add
    repeated_vals = list(
        set(val for val in to_check if val in seen or seen_add(val))
    )
    if not repeated_vals:
        repeats = False
        repeated_vals = None
    else:
        repeats = True

    return repeats, repeated_vals


def check_non_repeated_vals_lists(
    lists: list[list], allow_non_repeated: bool = True
) -> bool:
    """Checks that each list in a list of lists contains values which also
    occur in each and every other list.

    PARAMETERS
    ----------
    lists : list[lists]
    -   Master list containing the lists whose values should be checked for
        non-repeating values.

    allow_non_repeated : bool; default True
    -   Whether or not to allow non-repeated values to be present. If not, an
        error is raised if a non-repeated value is detected.

    RETURNS
    -------
    all_repeated : bool
    -   Whether or not all values of the lists are present in each and every
        other list.

    RAISES
    ------
    MissingEntryError
    -   Raised if a list contains a value that does not occur in each and every
        other list and 'allow_non_repeated' is 'False'.
    """

    compare_list = lists[0]
    all_repeated = True
    checking = True
    while checking:
        for check_list in lists[1:]:
            non_repeated_vals = [
                val for val in compare_list if val not in check_list
            ]
            non_repeated_vals.extend(
                [val for val in check_list if val not in compare_list]
            )
            if non_repeated_vals:
                if not allow_non_repeated:
                    raise MissingEntryError(
                        "Error when checking whether all values of a list are "
                        "repeated in another list:\nThe value(s) "
                        f"{non_repeated_vals} is(are) not present in all "
                        "lists.\n"
                    )
                else:
                    all_repeated = False
                    checking = False
        checking = False

    return all_repeated


def check_matching_entries(objects: list) -> bool:
    """Checks whether the entries of objects match one another.

    PARAMETERS
    ----------
    objects : list
    -   The objects whose entries should be compared.

    RETURNS
    -------
    matching : bool
    -   If True, the entries of the objects match. If False, the entries do not
        match.

    RAISES
    ------
    EntryLengthError
    -   Raised if the objects do not have equal lengths.
    """

    equal, length = check_lengths_list_identical(objects)
    if not equal:
        raise EntryLengthError(
            "Error when checking whether the entries of objects are "
            f"identical:\nThe lengths of the objects ({length}) do not "
            "match."
        )

    checking = True
    matching = True
    while checking and matching:
        object_i = 1
        for entry_i, base_value in enumerate(objects[0]):
            for object_values in objects[1:]:
                object_i += 1
                if object_values[entry_i] != base_value:
                    matching = False
                    checking = False
        checking = False

    return matching


def check_master_entries_in_sublists(
    master_list: list,
    sublists: list[list],
    allow_duplicates: bool = True,
) -> tuple[bool, Optional[list]]:
    """Checks whether all values in a master list are present in a set of
    sublists.

    PARAMETERS
    ----------
    master_list : list
    -   A master list of values.

    sublists : list[list]
    -   A list of sublists of values.

    allow_duplicates : bool; default True
    -   Whether or not to allow duplicate values to be present in the sublists.

    RETURNS
    -------
    all_present : bool
    -   Whether all values in the master list were present in the sublists.

    absent_entries : list | None
    -   The entry/ies of the master list missing from the sublists. If no
        entries are missing, this is None.
    """

    combined_sublists = list(chain(*sublists))

    if not allow_duplicates:
        duplicates, duplicate_entries = check_repeated_vals(combined_sublists)
        if duplicates:
            raise DuplicateEntryError(
                "Error when checking the presence of master list entries "
                f"within a sublist:\nThe entries {duplicate_entries} are "
                "repeated within the sublists.\nTo ignore this error, set "
                "'allow_duplicates' to True."
            )

    all_present = True
    absent_entries = []
    for entry in master_list:
        if entry not in combined_sublists:
            all_present = False
            absent_entries.append(entry)
    if not absent_entries:
        absent_entries = None

    return all_present, absent_entries


def check_sublist_entries_in_master(
    master_list: list,
    sublists: list[list],
    allow_duplicates: bool = True,
) -> tuple[bool, Optional[list]]:
    """Checks whether all values in a set of sublists are present in a master
    list.

    PARAMETERS
    ----------
    master_list : list
    -   A master list of values.

    sublists : list[list]
    -   A list of sublists of values.

    allow_duplicates : bool; default True
    -   Whether or not to allow duplicate values to be present in the sublists.

    RETURNS
    -------
    all_present : bool
    -   Whether all values in the sublists were present in the master list.

    absent_entries : list | None
    -   The entry/ies of the sublists missing from the master list. If no
        entries are missing, this is None.
    """

    combined_sublists = list(chain(*sublists))

    if not allow_duplicates:
        duplicates, duplicate_entries = check_repeated_vals(combined_sublists)
        if duplicates:
            raise DuplicateEntryError(
                "Error when checking the presence of master list entries "
                f"within a sublist:\nThe entries {duplicate_entries} are "
                "repeated within the sublists.\nTo ignore this error, set "
                "'allow_duplicates' to True."
            )

    all_present = True
    absent_entries = []
    for entry in combined_sublists:
        if entry not in master_list:
            all_present = False
            absent_entries.append(entry)
    if not absent_entries:
        absent_entries = None

    return all_present, absent_entries


def ordered_list_from_dict(list_order: list[str], dict_to_order: dict) -> list:
    """Creates a list from entries in a dictionary, sorted based on a given
    order.

    PARAMETERS
    ----------
    list_order : list[str]
    -   The names of keys in the dictionary, in the order that
        the values will occur in the list.

    dict_to_order : dict
    -   The dictionary whose entries will be added to the list.

    RETURNS
    -------
    list
    -   The ordered list.
    """

    return [dict_to_order[key] for key in list_order]


def ordered_dict_keys_from_list(
    dict_to_order: dict, keys_order: list[str]
) -> dict:
    """Reorders a dictionary so that the keys occur in a given order.

    PARAMETERS
    ----------
    dict_to_order : dict
    -   The dictionary to be ordered.

    keys_order : list[str]
    -   The order in which the keys should occur in the ordered dictionary.

    RETURNS
    -------
    ordered_dict : dict
    -   The dictionary with keys in a given order.
    """

    ordered_dict = {}
    for key in keys_order:
        ordered_dict[key] = dict_to_order[key]

    return ordered_dict


def check_if_ragged(
    to_check: Union[
        list[Union[list, NDArray]],
        NDArray,
    ]
) -> bool:
    """Checks whether a list or array of sublists or subarrays is 'ragged' (i.e.
    has sublists or subarrays with different lengths).

    PARAMETERS
    ----------
    to_check : list[list | numpy array] | numpy array[list | numpy array]
    -   The list or array to check.

    RETURNS
    -------
    ragged : bool
    -   Whether or not 'to_check' is ragged.
    """

    identical, _ = check_lengths_list_identical(to_check=to_check)
    if identical:
        ragged = False
    else:
        ragged = True

    return ragged


def ragged_array_to_list(
    ragged_array: NDArray,
) -> list[list]:
    """Converts a ragged numpy array of nested arrays to a ragged list of nested
    lists.

    PARAMETERS
    ----------
    ragged_array : numpy array[numpy array]
    -   The ragged array to convert to a list.

    RETURNS
    -------
    ragged_list : list[list]
    -   The ragged array as a list.
    """

    ragged_list = []
    for array in ragged_array:
        ragged_list.append(array.tolist())

    return ragged_list


def drop_from_list(obj: list, drop: list[str]) -> list:
    """Drops specified entries from a list.

    PARAMETERS
    ----------
    obj : list
    -   List with entries that should be dropped.

    drop : list
    -   List of entries to drop.

    RETURNS
    -------
    new_obj : list
    -   List with specified entries dropped.
    """

    new_obj = []
    for item in obj:
        if item not in drop:
            new_obj.append(item)

    return new_obj


def drop_from_dict(obj: dict, drop: list[str]) -> dict:
    """Removes specified entries from a dictionary.

    PARAMETERS
    ----------
    obj : dict
    -   Dictionary with entries to remove.

    drop : list[str]
    -   Names of the entries to remove.

    RETURNS
    -------
    new_obj : dict
    -   Dictionary with entries removed.
    """

    new_obj = deepcopy(obj)
    for item in drop:
        del new_obj[item]

    return new_obj


def _check_dimensions_results(
    dimensions: list[Union[str, list[str]]], results_key: str
) -> list[Union[str, list[str]]]:
    """Checks whether dimensions of results are in the correct format.

    PARAMETERS
    ----------
    dimensions : list[str] | list[list[str]]
    -   Dimensions of results, either a list of strings corresponding to the
        dimensions of all nodes/channels in the results, or a list of lists of
        strings, where each dimension corresponds to an individual node/channel.
    -   In the latter case, the dimensions of individual nodes/channels should
        not contain the axis "channel", as this is already the case. The
        "channels" axis will be set to the 0th axis, followed by the
        "frequencies" axis, followed by any other axes.
    -   E.g. if two channels were present, dimensions could be ["channels",
        "frequencies", "epochs", "timepoints"] or [["frequencies", "epochs",
        "timepoints"], ["timepoints", "frequencies", "epochs"]]. In the former
        case, the dimensions would be taken as-is. In the latter case,
        "channels" would be set to the 0th axis, and "frequencies" to the 1st
        axis, followed by any additional axes based on the order in which they
        occur in the first sublist, resulting in dimensions for all
        nodes/channels of ["channels", "frequencies", "epochs", "timepoints"].

    results_key : str
    -   Name of the entry in the results the dimensions are for.

    RETURNS
    -------
    dimensions : list[str] | list[list[str]]
    -   Dimensions of the results.
    -   If 'dimensions' was a list of strings, 'dimensions' is unchanged.
    -   If 'dimensions' was a list of sublists of strings in which each sublist
        of strings was the same, 'dimensions' is reduced to a single list of
        strings in which "channels" is set to the 0th axis.
    -   If 'dimensions' was a list of sublists of strings and each sublist of
        strings was not the same, 'dimensions' remains as a list of sublists.

    dims_to_find : list[str]
    -   Names of the dimensions and the order in which they should occur. If
        dimensions are given for each individual channel/node, the 0th axis is
        set to "frequencies", with the following axes set to the order in which
        they occur in the first node/channel.

    RAISES
    ------
    ValueError
    -   Raised if the dimensions are a list of sublists corresponding to the
        dimensions of individual channels/nodes, but with "channels" already
        being included in the dimensions of these individual channels/nodes
        which is, by their very nature, incorrect.
    """

    identical_dimensions = True
    if all(isinstance(entry, list) for entry in dimensions):
        for dims in dimensions:
            if "channels" in dims:
                raise ValueError(
                    "Error when trying to sort the dimensions of the results:\n"
                    "Multiple dimensions for the results entry "
                    f"'{results_key}' are present. In this case, it is assumed "
                    "that each entry in the dimensions corresponds to each "
                    "channel/node in the results. As a result, a 'channels' "
                    "axis should not be present in the dimensions, but it is "
                    f"{dims}.\nEither provide dimensions which are only a "
                    "single list of strings that applies to all channels, or "
                    "give dimensions for each individual node/channel (in "
                    "which case no 'channel' axis should be present in the "
                    "dimensions)."
                )
        identical_dimensions, _ = check_vals_identical(to_check=dimensions)
        if identical_dimensions:
            dimensions = ["channels", *dimensions[0]]
        else:
            check_non_repeated_vals_lists(
                lists=dimensions, allow_non_repeated=False
            )

    if identical_dimensions:
        dims_to_find = ["channels", "frequencies"]
        [
            dims_to_find.append(dim)
            for dim in dimensions
            if dim not in dims_to_find
        ]
    else:
        dims_to_find = ["frequencies"]
        [
            dims_to_find.append(dim)
            for dim in dimensions[0]
            if dim not in dims_to_find
        ]

    return dimensions, dims_to_find


def _sort_dimensions_results(results: dict, verbose: bool) -> tuple[dict, list]:
    """Rearranges the dimensions of attributes in a results dictionary so that
    the 0th axis corresponds to results from different channels, and the 1st
    dimension to different frequencies. If no dimensions, are given, the 0th
    axis is assumed to correspond to channels and the 1st axis to frequencies.
    -   Dimensions for an attribute, say 'X', would be containined in an
        attribute of the results dictionary under the name 'X_dimensions'.
    -   The dimensions should be provided as a list of strings containing the
        values 'channels' and 'frequencies' in the positions whose index
        corresponds to these axes in the values of 'X'. A single list should be
        given, i.e. 'X_dimensions' should hold for all entries of 'X'.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        ['epochs', 'channels', 'frequencies', 'timepoints'], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions ["channels", "frequencies", "epochs", "timepoints"].
    -   The axis for channels should be indicated as "channels", and the axis
        for frequencies should be marked as "frequencies".
    -   If the dimensions is a list of lists of strings, there should be a
        sublist for each channel/node in the results. Dimensions in the sublists
        should correspond to the results of each individual channel/node (i.e.
        no "channel" axis should be present in the dimensions of an individual
        node/channel as this is agiven).

    PARAMETERS
    ----------
    results : dict
    -   The results with dimensions of attributes to rearrange.

    verbose : bool
    -   Whether or not to report changes to the dimensions.

    RETURNS
    -------
    results : dict
    -   The results with dimensions of attributes in the appropriate order.

    dims_keys : list[str] | empty list
    -   Names of the dimension attributes in the results dictionary, or an empty
        list if no attributes are given.
    """

    dims_keys = []
    for key in results.keys():
        dims_key = f"{key}_dimensions"
        new_dims_set = False
        if dims_key in results.keys():
            dimensions, dims_to_find = _check_dimensions_results(
                dimensions=results[dims_key], results_key=key
            )
            if all(isinstance(entry, list) for entry in dimensions):
                for node_i, dims in enumerate(dimensions):
                    curr_axes_order = np.arange(len(dims)).tolist()
                    new_axes_order = [dims.index(dim) for dim in dims_to_find]
                    if new_axes_order != curr_axes_order:
                        results.at[node_i, key] = np.transpose(
                            results[key][node_i],
                            new_axes_order,
                        ).tolist()
                new_dims = ["channels", *dims_to_find]
                if verbose:
                    print(
                        f"Changing the dimensions of '{key}' which were "
                        "variable across the nodes/channels to a single "
                        f"dimension {new_dims}.\n"
                    )
            else:
                curr_axes_order = np.arange(len(dimensions)).tolist()
                new_axes_order = [dimensions.index(dim) for dim in dims_to_find]
                if new_axes_order != curr_axes_order:
                    results[key] = np.transpose(
                        results[key],
                        new_axes_order,
                    ).tolist()
                    new_dims_set = True
                old_dims = deepcopy(dimensions)
                new_dims = [dimensions[i] for i in new_axes_order]
                if verbose and new_dims_set:
                    print(
                        f"Rearranging the dimensions of '{key}' from "
                        f"{old_dims} to {new_dims}.\n"
                    )
            results[dims_key] = new_dims[1:]
            dims_keys.append(dims_key)

    return results, dims_keys


def _check_entry_lengths_results(
    results: dict, ignore: Union[list[str], None]
) -> int:
    """Checks that the lengths of list and numpy array entries in 'results' have
    the same length of axis 0.

    PARAMETERS
    ----------
    results : dict
    -   The results whose entries will be checked.

    ignore : list[str] | None
    -   The entries in 'results' which should be ignored, such as those which
        are identical across channels and for which only one copy is present.
        These entries are not included when checking the lengths, as these will
        be handled later.

    RETURNS
    -------
    length : int
    -   The lenghts of the 0th axis of lists and numpy arrays in 'results'.

    RAISES
    ------
    TypeError
    -   Raised if the 'results' contain an entry that is neither a list, numpy
        array, or dictionary.

    EntryLengthError
    -   Raised if the list or numpy array entries in 'results' do not all have
        the same length along axis 0.
    """

    if ignore is None:
        ignore = []

    supported_dtypes = [list, NDArray, dict]
    check_len_dtypes = [list, NDArray]

    to_check = []

    for key, value in results.items():
        if key not in ignore:
            dtype = type(value)
            if dtype in supported_dtypes:
                if dtype in check_len_dtypes:
                    to_check.append(value)
            else:
                raise TypeError(
                    "Error when trying to process the results:\nThe results "
                    f"dictionary contains an entry ('{key}') that is not of a "
                    f"supported data type ({supported_dtypes}).\n"
                )

    identical, length = check_lengths_list_identical(to_check=to_check, axis=0)
    if not identical:
        raise EntryLengthError(
            "Error when trying to process the results:\nThe length of "
            "entries in the results along axis 0 is not identical, but "
            "should be.\n"
        )

    return length


def _add_desc_measures_results(results: dict, entry_length: int) -> dict:
    """Adds descriptive processing measures to a results dictionary that can be
    updated as the results are processed.
    -   The entry 'n_from' is added, with the default value set to 1 (i.e. the
        result is derived from a sample size of n = 1). As results are e.g.
        averaged, 'n_from' will be updated to reflect the new sample size the
        results are derived from.

    PARAMETERS
    ----------
    results : dict
    -   Dictionary of results.

    entry_length : int
    -   Length of the 'n_from' list to add, which should match the number of
        channels/nodes of results in the dictionary.

    RETURNS
    -------
    results : dict
    -   Dictionary of results with entries for descriptive processing measures
        added.
    """

    results["n_from"] = [1] * entry_length

    return results


def _sort_identical_entries_results(
    results: dict,
    identical_entries: list[str],
    entry_length: int,
    verbose: bool,
) -> dict:
    """Creates a list equal to the length of other entries in 'results' for all
    entries specified in 'identical_entries', where each element of the list is
    a copy of the specified entries.

    PARAMETERS
    ----------
    results : dict
    -   The results dictionary with identical entries to sort.

    identical_entries : list[str]
    -   The entries in 'results' to convert to a list with length of axis 0
        equal to that of the 0th axis of other entries.

    entry_length : int
    -   The length of the 0th axis of entries in 'results'.

    verbose : bool
    -   Whether or not to print a description of the sorting process.

    RETURNS
    -------
    results : dict
    -   The results dictionary with identical entries sorted.
    """

    for entry in identical_entries:
        results[entry] = [deepcopy(results[entry])] * entry_length

    if verbose:
        print(
            f"Creating lists of the entries {identical_entries} in the results "
            f"with length {entry_length}.\n"
        )

    return results


def _add_dict_entries_to_results(
    results: dict, extract: dict[list[str]], entry_length: int, verbose: bool
) -> dict:
    """Extracts entries from dictionaries in 'results' and adds them to the
    results as a list whose length matches that of the other 'results' entries
    which are lists or numpy arrays.

    PARAMETERS
    ----------
    results : dict
    -   The results containing the dictionaries whose values should be
        extracted.

    extract : dict[list[str]]
    -   Dictionary whose keys are the names of dictionaries in 'results', and
        whose values are a list of strings corresponding to the entries in the
        dictionaries in 'results' to extract.

    entry_length : int
    -   The length of the 0th axis of entries in 'results'.

    verbose : bool
    -   Whether or not to print a description of the sorting process.

    RETURNS
    -------
    results : dict
    -   The results with the desired dictionary entries extracted.
    """

    for dict_name, dict_entries in extract.items():
        for entry in dict_entries:
            if entry in results.keys():
                raise PreexistingAttributeError(
                    f"Error when processing the results:\nThe entry '{entry}' "
                    f"from the dictionary '{dict_name}' is being extracted and "
                    "added to the results, however an attribute named "
                    f"'{entry}' is already present in the results.\n"
                )
            repeat_val = results[dict_name][entry]
            if isinstance(repeat_val, dict):
                raise TypeError(
                    "Error when processing the results:\nThe results contain "
                    f"the dictionary '{dict_name}' which contains an entry "
                    f"'{entry}' that is being extracted and included with the "
                    "results for processing, however processing dictionaries "
                    "is not supported.\n"
                )
            results[entry] = [deepcopy(repeat_val)] * entry_length

        if verbose:
            print(
                f"Extracting the entries {dict_entries} from the dictionary "
                f"'{dict_name}' into the results with length {entry_length}.\n"
            )

    return results


def _drop_dicts_from_results(results: dict) -> dict:
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


def _sort_dicts_results(
    results: dict,
    extract_from_dicts: Union[dict[list[str]], None],
    entry_length: int,
    verbose: bool,
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

    verbose : bool
    -   Whether or not to print a description of the sorting process.

    RETURNS
    -------
    dict
    -   The sorted results, with the desired dictionary entries extracted, if
        applicable, and the dictionaries discarded.
    """

    if extract_from_dicts is not None:
        results = _add_dict_entries_to_results(
            results=results,
            extract=extract_from_dicts,
            entry_length=entry_length,
            verbose=verbose,
        )

    return _drop_dicts_from_results(results=results)


def sort_inputs_results(
    results: dict,
    extract_from_dicts: Union[dict[list[str]], None],
    identical_entries: Union[list[str], None],
    discard_entries: Union[list[str], None],
    verbose: bool = True,
) -> None:
    """Checks that the values in 'results' are in the appropriate format for
    processing with PostProcess or Plotting class objects.

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
        results = drop_from_dict(obj=results, drop=discard_entries)

    results, dims_keys = _sort_dimensions_results(
        results=results, verbose=verbose
    )

    if identical_entries is None:
        identical_entries = []
    identical_entries = [*identical_entries, *dims_keys]

    entry_length = _check_entry_lengths_results(
        results=results, ignore=identical_entries
    )

    results = _add_desc_measures_results(
        results=results, entry_length=entry_length
    )

    if identical_entries is not None:
        results = _sort_identical_entries_results(
            results=results,
            identical_entries=identical_entries,
            entry_length=entry_length,
            verbose=verbose,
        )

    results = _sort_dicts_results(
        results=results,
        extract_from_dicts=extract_from_dicts,
        entry_length=entry_length,
        verbose=verbose,
    )

    return results


def dict_to_df(obj: dict) -> pd.DataFrame:
    """Converts a dictionary into a pandas DataFrame.

    PARAMETERS
    ----------
    obj : dict
    -   Dictionary to convert.

    RETURNS
    -------
    pandas DataFrame
    -   The converted dictionary.
    """

    return pd.DataFrame.from_dict(data=obj, orient="columns")
