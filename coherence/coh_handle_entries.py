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
"""

from itertools import chain
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from coh_exceptions import (
    DuplicateEntryError,
    EntryLengthError,
    MissingEntryError,
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


def drop_from_list(obj: list, drop: list) -> list:
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
