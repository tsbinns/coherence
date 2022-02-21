"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""




from abc import ABC, abstractmethod




ProcMethod(ABC):
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
    process(self) -> None:
        """Performs the processing on the data."""

    
    @abstractmethod
    save(self,
        data: pd.DataFrame,
        fpath: str,
        ftype: str,
        verbose : bool = True,
        ) -> None:
        """Saves the processed data to a specified location as a specified
        filetype.

        data : pd.DataFrame
        -   The data to be saved.

        fpath : str
        -   The filepath where the data should be saved.

        ftype : str
        -   The filetype which the data should be saved as. Must be a filetype
            which a pd.DataFrame object can be saved as.

        verbose : bool | default True
        -   Whether or not the user will be warned if they are about to
            overwrite a pre-existing file. If True (default), the user is
            warned and prompted to continue with or abort the saving procedure.
            If False, the user is not warned and any pre-existing conflicting
            files are overwritten.
        """


