"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union
import numpy as np
import coh_signal


class ProcMethod(ABC):
    """Abstract class for implementing data processing methods.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   A preprocessed Signal object whose data will be processed.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

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
    -   Performs power analysis on data using Morlet wavelets.

    PowerFOOOF
    -   Performs power analysis on data using FOOOF.

    ConnectivityCoh
    -   Performs connectivity analysis on data with coherence.

    ConnectivityiCoh
    -   Performs connectivity analysis on data with the imaginary part of
        coherence.
    """

    @abstractmethod
    def __init__(self, signal: coh_signal.Signal, verbose: bool) -> None:

        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self.processing_steps = None
        self.extra_info = None

        # Initialises inputs of the ProcMethod object.
        self.signal = deepcopy(signal)
        self._verbose = verbose

        # Initialises aspects of the ProcMethod object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._processed = False

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""

    @abstractmethod
    def _sort_inputs(self):
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing and assigns inputs."""

        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.extra_info = deepcopy(self.signal.extra_info)

    def _prepare_results_for_saving(
        self,
        results: np.array,
        results_dims: Union[list[str], None],
        rearrange: Union[list[str], None],
    ) -> list:
        """Copies analysis results and rearranges their dimensions as specified
        in preparation for saving.

        PARAMETERS
        ----------
        results : numpy array
        -   The results of the analysis.

        results_dims : list[str] | None; default None
        -   The names of the axes in the results, used for rearranging the axes.
            If None, the data cannot be rearranged.

        rearrange : list[str] | None; default None
        -   How to rearrange the axes of the data once extracted. If given,
            'results_structure' must also be given.
        -   E.g. ["channels", "epochs", "timepoints"] would give data in the
            format channels x epochs x timepoints
        -   If None, the data is taken as is.

        RETURNS
        -------
        extracted_results : array
        -   The transformed results.
        """

        extracted_results = deepcopy(results)

        if rearrange:
            extracted_results = np.transpose(
                extracted_results,
                [results_dims.index(axis) for axis in rearrange],
            )

        return extracted_results.tolist()
