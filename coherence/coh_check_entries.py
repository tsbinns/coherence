from abc import ABC, abstractmethod
from typing import Any, Union



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
        """Finds the lengths of the entries in the input object.
        -   Implemented in the subclasses' method.
        """
        pass


    @abstractmethod
    def identical(self,
        entry_lengths: list[int]
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

        lengths : int or list[int]
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
    def equals_n(self,
        entry_lengths: list[int],
        n: int
        ) -> bool:
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
    -   Subclass of the abstract class CheckLengths.

    PARAMETERS
    ----------
    to_check : dict[Any, Any]
    -   The dictionary for which the lengths of the entries should be checked.

    ignore_values : list[Any] | optional, default []
    -   The values of entries within 'to_check' to ignore when checking the 
        lengths of entries.
    -   If [] (default), no entries are ignored.

    ignore_keys : list[Any] | optional, default []
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

    def __init__(self,
        to_check: dict[Any, Any],
        ignore_values: list[Any] = [],
        ignore_keys: list[Any] = []
        ) -> None:

        self.to_check = to_check
        self.ignore_values = ignore_values
        self.ignore_keys = ignore_keys


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
        -   Partially implemented in the parent class' method.

        RETURNS
        -------
        bool
        -   Whether or not the lengths of the entries are identical.

        int or list
        -   The length(s) of the entries. If the lengths are identical,
            'lengths' is an int representing the length of all items.
        -   If the lengths are not identical, 'lengths' is a list containing the
            lengths of the individual entries (i.e. 'entry_lengths').
        """

        self._check()

        return super().identical(self.entry_lengths)


    def check_equals_n(self,
        n: int
        ) -> bool:
        """Checks whether the lengths of entries within the input object are
        equal to a given integer.
        -   Partially implemented in the parent class' method.

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
    -   Subclass of the abstract class CheckLengths.

    PARAMETERS
    ----------
    to_check : list[Any]
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list[Any] | optional, default []
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

    def __init__(self,
        to_check: list[Any],
        ignore_values: list[Any] = []
        ) -> None:

        self.to_check = to_check
        self.ignore_values = ignore_values


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
        -   Partially implemented in the parent class' method.

        RETURNS
        -------
        bool
        -   Whether or not the lengths of the entries are identical.

        int or list
        -   The length(s) of the entries. If the lengths are identical,
            'lengths' is an int representing the length of all items.
        -   If the lengths are not identical, 'lengths' is a list containing the
            lengths of the individual entries (i.e. 'entry_lengths').
        """

        self._check()

        return super().identical(self.entry_lengths)


    def equals_n(self,
        n: int
        ) -> bool:
        """Checks whether the lengths of entries within the input object are
        equal to a given integer.
        -   Partially implemented in the parent class.

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


