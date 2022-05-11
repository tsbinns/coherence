"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
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
        self.results = None
        self._results_dims = None
        self.processing_steps = None
        self.extra_info = None

        # Initialises inputs of the ProcMethod object.
        self.signal = deepcopy(signal)
        self._verbose = verbose

        # Initialises aspects of the ProcMethod object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._processed = False
        self._windows_averaged = False

    @abstractmethod
    def process(self) -> None:
        """Performs the processing on the data."""

    @abstractmethod
    def _sort_inputs(self) -> None:
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing and assigns inputs."""

        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.extra_info = deepcopy(self.signal.extra_info)

    @property
    def results_dims(self) -> list[str]:
        """Returns the dimensions of the results, corresponding to the results
        that will be returned with the 'get_results' method.

        RETURNS
        -------
        dims : list[str]
        -   Dimensions of the results.
        """

        if self._windows_averaged:
            dims = self._results_dims[1:]
        else:
            dims = self._results_dims

        return deepcopy(dims)

    @abstractmethod
    def save_results(self) -> None:
        """"""

    def _get_optimal_dims(self) -> tuple[list, list[str]]:
        """Finds the optimal order of dimensions for the results, following the
        order ["windows", "channels", "epochs", "frequencies", "timepoints"]
        based on which dimensions are present in the reuslts.

        RETURNS
        -------
        optimal_dims : list[str]
        -   Optimal dimensions of the results.
        """

        possible_order = [
            "windows",
            "channels",
            "epochs",
            "frequencies",
            "timepoints",
        ]
        optimal_dims = [
            dim for dim in possible_order if dim in self.results_dims
        ]

        return optimal_dims
