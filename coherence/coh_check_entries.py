from abc import ABC, abstractmethod



class CheckLengths(ABC):

    @abstractmethod
    def _check():
        pass



class CheckLengthsDict(CheckLengths):

    def _check(self,
        to_check: list,
        ignore_values: list,
        ignore_keys: list
        ) -> list:

        entry_lengths = []
        for key in self._to_check.keys():
            value = self._to_check[key]
            if key not in self._ignore_keys or value not in self._ignore_values:
                entry_lengths.append(len(value))

        return entry_lengths


    def check_identical(self,
        to_check: list,
        ignore_values: list = [],
        ignore_keys: list = []
        ) -> tuple[bool, int or list]:

        entry_lengths = self._check(to_check, ignore_values, ignore_keys)
        
        if all(entry_lengths) == entry_lengths[0]:
            to_return = [True, entry_lengths[0]]
        else:
            to_return = [False, entry_lengths]

        return to_return


    def check_equals_n(self,
        to_check: list,
        n: int,
        ignore_values: list = [],
        ignore_keys: list = []
        ) -> bool:

        entry_lengths = self._check(to_check, ignore_values, ignore_keys)
        
        if all(entry_lengths) == n:
            all_n = True
        else:
            all_n = False

        return all_n



class CheckLengthsList(CheckLengths):

    @abstractmethod
    def _check(self,
        to_check: list,
        ignore_values: list
        ) -> list:

        entry_lengths = []
        for value in to_check:
            if value not in ignore_values:
                entry_lengths.append(len(value))

        return entry_lengths



class CheckLengthsListIdentical(CheckLengthsList):

    def __init__(self,
        to_check: list,
        ignore_values: list = []
        ) -> None:

        self.to_check = to_check
        self.ignore_values = ignore_values


    def _check(self
        ) -> list:

        return super()._check(self.to_check, self.ignore_values)

    
    def check(self
        ) -> tuple[bool, int or list]:

        entry_lengths = self._check()
        
        if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
            to_return = [True, entry_lengths[0]]
        else:
            to_return = [False, entry_lengths]

        return to_return



class CheckLengthsListEqualsN(CheckLengthsList):

    def __init__(self,
        to_check: list,
        n: int,
        ignore_values: list = []
        ) -> None:

        self.to_check = to_check
        self.n = n
        self.ignore_values = ignore_values


    def _check(self
        ) -> list:

        return super()._check(self.to_check, self.ignore_values)

    
    def check(self
        ) -> bool:

        entry_lengths = self._check()

        if all(entry_lengths) == self.n:
            all_n = True
        else:
            all_n = False
            
        return all_n