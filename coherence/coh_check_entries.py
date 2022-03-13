"""Contains classes for checking properties of entries within objects.

CLASSES
-------
CheckLengths : abstract base class
-   Abstract class for checking the lengths of entries within an input
    object.

CheckLengthsDict : subclass of the abstract base class CheckLengths
-   Checks the lengths of entries within an input dictionary.

CheckLengthsList : subclass of the abstract base class CheckLengths
-   Checks the lengths of entries within an input list.

CheckDuplicates : abstract base class
-   Abstract class for checking whether duplicate entries exist within an input
    object.

CheckDuplicatesList : subclass of the abstract base class CheckDuplicates
-   Checks whether duplicates exist within an input list.

CheckMatchingEntries
-   Checks whether entries within two objects match.

CheckEntriesPresent
-   Checks whether entries of a sublist(s) are present within a master list and
    vice versa.
"""


from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Optional, Union

from coh_exceptions import DuplicateEntryError, EntryLengthError


class CheckLengths(ABC):
    """Abstract class for checking the lengths of entries within an input
    object.

    METHODS
    -------
    identical (abstract)
    -   Checks whether the lengths of the entries are the same.

    equals_n (abstract)
    -   Checks whether the lengths of the entries are equal to a given integer.

    SUBCLASSES
    ----------
    CheckLengthsDict
    -   Checks whether the lengths of entries in a dictionary are the same.

    CheckLengthsList
    -   Checks whether the lengths of entries within a list are the same.
    """

    @abstractmethod
    def _check(self):
        """Finds the lengths of the entries in the input object."""

    @abstractmethod
    def identical(
        self, entry_lengths: list[int]
    ) -> tuple[bool, Union[int, list[int]]]:
        """Checks whether the lengths of entries in the input object are
        identical.

        PARAMETERS
        ----------
        entry_lengths : list[int]
        -   List containing the lengths of entries within the input object.

        RETURNS
        -------
        identical : bool
        -   Whether or not the lengths of the entries are identical.

        lengths : int | list[int]
        -   The length(s) of the entries. If the lengths are identical,
            'lengths' is an int representing the length of all items.
        -   If the lengths are not identical, 'lengths' is a list containing the
            lengths of the individual entries (i.e. 'entry_lengths').
        """

        if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
            identical = True
            lengths = entry_lengths[0]
        else:
            identical = False
            lengths = entry_lengths

        return identical, lengths

    @abstractmethod
    def equals_n(self, entry_lengths: list[int], n: int) -> bool:
        """Checks whether the lengths of entries within the input object are
        equal to a given integer.

        PARAMETERS
        ----------
        entry_lengths : list[int]
        -   List containing the lengths of entries within the input object.

        n : int
        -   The integer which the lengths of the entries should be equal to.

        RETURNS
        -------
        all_n : bool
        -   Whether or not the lengths of the entries are equal to 'n'.
        """

        if entry_lengths.count(n) == len(entry_lengths):
            all_n = True
        else:
            all_n = False

        return all_n


class CheckLengthsDict(CheckLengths):
    """Checks the lengths of entries within an input dictionary.

    PARAMETERS
    ----------
    to_check : dict[Any, Any]
    -   The dictionary for which the lengths of the entries should be checked.

    ignore_values : list[Any]; optional, default []
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If [] (default), no entries are ignored.

    ignore_keys : list[Any]; optional, default []
    -   The keys of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If [] (default), no entries are ignored.

    METHODS
    -------
    identical
    -   Checks whether the lengths of the entries are the same.

    equals_n
    -   Checks whether the lengths of the entries are equal to a given integer.
    """

    def __init__(
        self,
        to_check: dict[Any, Any],
        ignore_values: list[Any] = [],
        ignore_keys: list[Any] = [],
    ) -> None:

        # Initialises inputs of the object.
        self.to_check = to_check
        self.ignore_values = ignore_values
        self.ignore_keys = ignore_keys

        # Initialises aspects of the object that will later be filled.
        self.entry_lengths = None
        self.n = None

    def _check(self) -> None:
        """Finds the lengths of the values of the entries within the input
        dictionary.
        """

        self.entry_lengths = []
        for key, value in self.to_check.items():
            if key not in self.ignore_keys or value not in self.ignore_values:
                self.entry_lengths.append(len(value))

    def identical(self) -> tuple[bool, Union[int, list[int]]]:
        """Checks whether the lengths of entries in the input dictionary are
        identical.

        RETURNS
        -------
        bool
        -   Whether or not the lengths of the entries are identical.

        int | list
        -   The length(s) of the entries. If the lengths are identical,
            'lengths' is an int representing the length of all items.
        -   If the lengths are not identical, 'lengths' is a list containing the
            lengths of the individual entries (i.e. 'entry_lengths').
        """

        self._check()

        return super().identical(self.entry_lengths)

    def check_equals_n(self, n: int) -> bool:
        """Checks whether the lengths of entries within the input object are
        equal to a given integer.

        PARAMETERS
        ----------
        n : int
        -   The integer which the lengths of the entries should be equal to.

        RETURNS
        -------
        bool
        -   Whether or not the lengths of the entries are equal to 'n'.
        """

        self.n = n
        self._check()

        return super().equals_n(self.entry_lengths, self.n)


class CheckLengthsList(CheckLengths):
    """Checks the lengths of entries within an input list.

    PARAMETERS
    ----------
    to_check : list[Any]
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list[Any]; optional, default []
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If [] (default), no entries are ignored.

    METHODS
    -------
    identical
    -   Checks whether the lengths of the entries in are the same.

    equals_n
    -   Checks whether the lengths of the entries are equal to a given integer.
    """

    def __init__(
        self, to_check: list[Any], ignore_values: list[Any] = []
    ) -> None:

        # Initialises inputs of the object.
        self.to_check = to_check
        self.ignore_values = ignore_values

        # Initialises aspects of the object that will later be filled.
        self.entry_lengths = None
        self.n = None

    def _check(self) -> None:
        """Finds the lengths of the values of the entries within the input
        dictionary.
        """

        self.entry_lengths = []
        for value in self.to_check:
            if value not in self.ignore_values:
                self.entry_lengths.append(len(value))

    def identical(self) -> tuple[bool, Union[int, list[int]]]:
        """Checks whether the lengths of entries in the input dictionary are
        identical.

        RETURNS
        -------
        bool
        -   Whether or not the lengths of the entries are identical.

        int | list
        -   The length(s) of the entries. If the lengths are identical,
            'lengths' is an int representing the length of all items.
        -   If the lengths are not identical, 'lengths' is a list containing the
            lengths of the individual entries (i.e. 'entry_lengths').
        """

        self._check()

        return super().identical(self.entry_lengths)

    def equals_n(self, n: int) -> bool:
        """Checks whether the lengths of entries within the input object are
        equal to a given integer.

        PARAMETERS
        ----------
        n : int
        -   The integer which the lengths of the entries should be equal to.

        RETURNS
        -------
        bool
        -   Whether or not the lengths of the entries are equal to 'n'.
        """

        self.n = n
        self._check()

        return super().equals_n(self.entry_lengths, self.n)


class CheckDuplicates(ABC):
    """Abstract class for checking whether duplicate entries exist within an
    input object.

    METHODS
    -------
    check (abstract)
    -   Checks whether duplicates are present in the object.

    SUBCLASSES
    ----------
    CheckDuplicatesList
    -   Checks whether duplicates exist within an input list.
    """

    @abstractmethod
    def check(self):
        """Finds and returns duplicates within an input object."""


class CheckDuplicatesList(CheckDuplicates):
    """Checks whether duplicates exist within an input list.

    PARAMETERS
    ----------
    values : list[Any]
    -   The list of values whose entries should be checked for duplicates.

    METHODS
    -------
    check
    -   Checks whether duplicates are present in the list.
    """

    def __init__(self, values: list[Any]) -> None:

        self.values = values

        self.check()

    def check(self) -> tuple[bool, Optional[list[Any]]]:
        """Checks to see if there are any duplicate values present in a list of
        values.

        RETURNS
        -------
        duplicates : bool
        -   Whether or not duplicates are present.

        duplicate_values : list[Any] | None
        -   The list of duplicates values, or None if no duplicates are present.
        """

        duplicates = False
        seen = set()
        seen_add = seen.add
        duplicate_values = list(
            set(
                value
                for value in self.values
                if value in seen or seen_add(value)
            )
        )
        if not duplicate_values:
            duplicate_values = None
        else:
            duplicates = True

        return duplicates, duplicate_values


class CheckMatchingEntries:
    """Checks whether the entries of two objects match one another.
    PARAMETERS
    ----------
    object_1 : Any
    -   The object whose entries should be checked.

    object_2 : Any
    -   The object whose entries object_1 should be compared against.

    RETURNS
    -------
    matching : bool
    -   If True, the entries of the objects match. If False, the entries do
        not match.
    """

    def __init__(self, object_1: Any, object_2: Any) -> bool:

        self.object_1 = object_1
        self.object_2 = object_2

        self._sort_inputs()
        self._check()

    def _sort_inputs(self) -> None:
        """Checks to make sure that the objects whose entries are being
        compared have equal length.

        RAISES
        ------
        EntryLengthError
        -   Raised if the two objects do not have equal lengths.
        """

        equal, length = CheckLengthsList(
            [self.object_1, self.object_2]
        ).identical()
        if not equal:
            raise EntryLengthError(
                "Error when checking whether the entries of two objects are "
                f"identical:\nThe lengths of the objects ({length}) do not "
                "match."
            )

    def _check(self) -> bool:
        """Checks whether the entries of the two input objects are identical.

        RETURNS
        -------
        matching : bool
        -   If True, the entries of the objects match. If False, the entries do
            not match.
        """

        matching = True

        while matching:
            for i, value in enumerate(self.object_1):
                if value != self.object_2[i]:
                    matching = False

        return matching


class CheckEntriesPresent:
    """Checks whether entries of a sublist(s) are present within a master list
    and vice versa.

    PARAMETERS
    ----------
    master_list : list[Any]
    -   A master list of values.

    sublists : list[list[Any]]
    -   A list of sublists of values.

    METHODS
    -------
    master_in_subs
    -   Checks whether all values in a master list are present in a set of
        sublists.

    subs_in_master
    -   Checks whether all values within a set of sublists are present in a
        master list.
    """

    def __init__(
        self, master_list: list[Any], sublists: list[list[Any]]
    ) -> None:

        self._master_list = master_list
        self._combined_sublists = list(chain(*sublists))

    def _check_for_duplicates(self) -> None:
        """Checks whether duplicate values are present within the sublists.

        RAISES
        ------
        DuplicateEntryError
        -   Raised if duplicate values are present within the sublists.
        """

        duplicates, duplicate_entries = CheckDuplicatesList(
            self._combined_sublists
        ).check()
        if duplicates:
            raise DuplicateEntryError(
                "Error when checking the presence of master list entries "
                f"within a sublist:\nThe entries {duplicate_entries} are "
                "repeated within the sublists.\nTo ignore this error, set "
                "'allow_duplicates' to True."
            )

    def master_in_subs(
        self, allow_duplicates: bool = True
    ) -> tuple[bool, Optional[list[Any]]]:
        """Checks whether all values in a master list are present in a set of
        sublists.

        PARAMETERS
        ----------
        allow_duplicates : bool, default True
        -   Whether or not to allow duplicate values to be present within the
            sublists.

        RETURNS
        -------
        all_present : bool
        -   Whether all values in the master list were present in the sublists.

        absent_entries : list[Any] | None
        -   The entry/ies of the master list missing from the sublists. If no
            entries are missing, this is None.
        """

        if not allow_duplicates:
            self._check_for_duplicates()

        all_present = True
        absent_entries = []
        for entry in self._master_list:
            if entry not in self._combined_sublists:
                all_present = False
                absent_entries.append(entry)
        if not absent_entries:
            absent_entries = None

        return all_present, absent_entries

    def subs_in_master(
        self, allow_duplicates: bool = True
    ) -> tuple[bool, Optional[list[Any]]]:
        """Checks whether all values in the sublists list are present in a set
        the master list.

        PARAMETERS
        ----------
        allow_duplicates : bool, default True
        -   Whether or not to allow duplicate values to be present within the
            sublists.

        RETURNS
        -------
        all_present : bool
        -   Whether all values in the sublists were present in the master list.

        absent_entries : list[Any] | None
        -   The entry/ies of the sublists missing from the master list. If no
            entries are missing, this is None.
        """

        if not allow_duplicates:
            self._check_for_duplicates()

        all_present = True
        absent_entries = []
        for entry in self._combined_sublists:
            if entry not in self._master_list:
                all_present = False
                absent_entries.append(entry)
        if not absent_entries:
            absent_entries = None

        return all_present, absent_entries
