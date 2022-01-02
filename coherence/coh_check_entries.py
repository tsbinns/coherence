from abc import ABC, abstractmethod



class CheckLengths(ABC):

    @abstractmethod
    def _check():
        pass


    @abstractmethod
    def identical(self,
        entry_lengths: list
        ) -> tuple(bool, int or list):

        if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
            to_return = [True, entry_lengths[0]]
        else:
            to_return = [False, entry_lengths]

        return to_return

    
    @abstractmethod
    def equals_n(self,
        entry_lengths: list,
        n: int
        ) -> bool:

        if all(entry_lengths) == n:
            all_n = True
        else:
            all_n = False
            
        return all_n



class CheckLengthsDict(CheckLengths):

    def __init__(self,
        to_check: list,
        ignore_values: list = [],
        ignore_keys: list = []
        ) -> None:

        self.to_check = to_check
        self.ignore_values = ignore_values
        self.ignore_keys = ignore_keys


    def _check(self
        ) -> list:

        self.entry_lengths = []
        for key, value in self.to_check.items():
            if key not in self.ignore_keys or value not in self.ignore_values:
                self.entry_lengths.append(len(value))


    def identical(self
        ) -> tuple[bool, int or list]:

        self._check()

        return super().identical(self.entry_lengths)


    def check_equals_n(self,
        n: int
        ) -> bool:

        self.n = n
        self._check()

        return super().equals_n(self.entry_lengths, self.n)



class CheckLengthsList(CheckLengths):

    def __init__(self,
        to_check: list,
        ignore_values: list = []
        ) -> None:

        self.to_check = to_check
        self.ignore_values = ignore_values


    def _check(self
        ) -> list:

        self.entry_lengths = []
        for value in self.to_check:
            if value not in self.ignore_values:
                self.entry_lengths.append(len(value))

    
    def identical(self
        ) -> tuple[bool, int or list]:

        self._check()

        return super().identical(self.entry_lengths)


    def equals_n(self,
        n: int
        ) -> bool:

        self.n = n
        self._check()

        return super().equals_n(self.entry_lengths, self.n)


