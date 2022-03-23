"""Contains functions for handling entries within objects.

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

check_duplicates_list
-   Checks whether duplicates exist within an input list.

check_matching_entries
-   Checks whether the entries of objects match one another.

check_master_entries_in_sublists
-   Checks whether all values in a master list are present in a set of sublists.

check_sublist_entries_in_master
-   Checks whether all values in a set of sublusts are present in a master list.

ordered_list_from_dict
-   Creates a list from entries in a dictionary, sorted based on a given order.
"""

from itertools import chain
from typing import Optional, Union
from coh_exceptions import DuplicateEntryError, EntryLengthError


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
    to_check: list, ignore_values: Optional[list]
) -> list[int]:
    """Finds the lengths of entries within a list.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

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
            entry_lengths.append(len(value))

    return entry_lengths


def check_lengths_list_identical(
    to_check: list, ignore_values: Optional[list] = None
) -> tuple[bool, Union[int, list[int]]]:
    """Checks whether the lengths of entries in the input dictionary are
        identical.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

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
        to_check=to_check, ignore_values=ignore_values
    )

    if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
        identical = True
        lengths = entry_lengths[0]
    else:
        identical = False
        lengths = entry_lengths

    return identical, lengths


def check_lengths_list_equals_n(
    to_check: list, n: int, ignore_values: Optional[list] = None
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

    RETURNS
    -------
    all_n : bool
    -   Whether or not the lengths of the entries are equal to 'n'.
    """

    entry_lengths = _find_lengths_list(
        to_check=to_check, ignore_values=ignore_values
    )

    if entry_lengths.count(n) == len(entry_lengths):
        all_n = True
    else:
        all_n = False

    return all_n


def check_duplicates_list(
    values: list,
) -> tuple[bool, Optional[list]]:
    """Checks whether duplicates exist within an input list.

    PARAMETERS
    ----------
    values : list
    -   The list of values whose entries should be checked for duplicates.

    RETURNS
    -------
    duplicates : bool
    -   Whether or not duplicates are present.

    duplicate_values : list | None
    -   The list of duplicates values, or None if no duplicates are present.
    """

    seen = set()
    seen_add = seen.add
    duplicate_values = list(
        set(value for value in values if value in seen or seen_add(value))
    )
    if not duplicate_values:
        duplicates = False
        duplicate_values = None
    else:
        duplicates = True

    return duplicates, duplicate_values


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
        duplicates, duplicate_entries = check_duplicates_list(combined_sublists)
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
        duplicates, duplicate_entries = check_duplicates_list(combined_sublists)
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
