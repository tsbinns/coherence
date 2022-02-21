"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""




from abc import ABC, abstractmethod
from os.path import exists




class ProcMethod(ABC):
    """Abstract class for implementing data processing methods.
    -   This class should not be called directly. Instead, its subclasses should
        be called from their respective files.

    METHODS
    -------
    process (abstract)
    -   Performs the processing on the data.

    save (abstract)
    -   Saves the processed data to a specified location as a specified
        filetype.

    SUBCLASSES
    ----------
    PowerMorlet
    -   Performs power analysis on the data using Morlet wavelets.

    ConnectivityCoh
    -   Performs connectivity analysis on the data using coherence as the
        measure.

    ConnectivityiCoh
    -   Performs connectivity analysis on the data using the imaginary part of
        coherence as the measure.
    """

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""

    
    @abstractmethod
    def save(self,
        fpath: str,
        obj: Any,
        attr_to_save: list[str],
        ask_before_overwrite: bool = True
        ) -> None:
        """Saves the processed data to a specified location.
        
        PARAMETERS
        ----------
        fpath : str
        -   The filepath for where to save the object.

        obj : Any
        -   The object to save.

        attr_to_save : list[str]
        -   The names of the attributes of the object to save.

        ask_before_overwrite : bool | Optional, default True
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists. If False, the user is not asked to
            confirm this and it is done automatically.
        """
        if exists(fpath):
            if ask_before_overwrite:
                write = False
                valid_response = False
                while valid_response is False:
                    response = input(
                        f"The file '{fpath}' already exists.\nDo you want to "
                        "overwrite it? y/n: "
                    )
                    if response not in ['y', 'n']:
                        print(
                            "The only accepted responses are 'y' and 'n'. "
                            "Please input your response again."
                        )
                        break
                    if response == 'n':
                        print(
                            "You have requested that the pre-existing file not "
                            "be overwritten. The new file has not been saved."
                        )
                        valid_response = True
                    if response == 'y':
                        write = True
                        valid_response = True
            else:
                write = True
        else:
            write = True

        if write:
            with open(fpath, 'wb') as file:
                pickle.dump(SaveObject(obj, attr_to_save), file)
