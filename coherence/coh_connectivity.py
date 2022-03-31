"""Classes for calculating connectivity between signals.

METHODS
-------
ConnectivityCoherence : subclass of the abstract base class 'ProcMethod'
-   Calculates the coherence (standard or imaginary) between signals.
"""

from typing import Optional, Union
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs
import numpy as np
from coh_exceptions import (
    EntryLengthError,
    InputTypeError,
    ProcessingOrderError,
)
from coh_handle_entries import check_lengths_list_equals_n
from coh_processing_methods import ProcMethod
import coh_signal


class ConnectivityCoherence(ProcMethod):
    """Calculates the coherence (standard or imaginary) between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs coherence analysis.

    save_object
    -   Saves the ConnectivityCoherence object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)

        # Initialises inputs of the ConnectivityCoherence object.
        self._sort_inputs()

        # Initialises aspects of the ConnectivityCoherence object that will be
        # filled with information as the data is processed.
        self.coherence = None
        self.coherence_dims = None
        """
        self.method = None
        self.indices = None
        self.mode = None
        self.fmin = None
        self.fmax = None
        self.fskip = None
        self.faverage = None
        self.tmin = None
        self.tmax = None
        self.mt_bandwidth = None
        self.mt_adaptive = None
        self.mt_low_bias = None
        self.cwt_freqs = None
        self.cwt_n_cycles = None
        self.block_size = None
        self.n_jobs = None
        """

    def _sort_inputs(self) -> None:
        """Checks the inputs to the PowerFOOOF object to ensure that they
        match the requirements for processing and assigns inputs.

        RAISES
        ------
        InputTypeError
        -   Raised if the PowerMorlet object input does not contain data in a
            supported format.
        """

        supported_data_dims = [["epochs", "channels", "timepoints"]]
        if self.signal.power_dims not in supported_data_dims:
            raise InputTypeError(
                "Error when performing coherence analysis on the data:\nThe "
                f"preprocessed data is in the form {self.signal.power_dims}, "
                f"but only data in the form {supported_data_dims} is supported."
            )

        super()._sort_inputs()

    def _sort_indices(
        self, indices: Union[dict[Union[str, list[str]]], None]
    ) -> None:
        """"""

        if indices is not None:
            two_entries = check_lengths_list_equals_n(
                to_check=indices.keys(), n=2
            )
            if not two_entries:
                raise EntryLengthError(
                    "Error when determining the channel indices for "
                    "calculating connectivity:\nThe indices dictionary should "
                    "have two entries corresponding to the seed channels and "
                    "target channels, however this is not the case."
                )

        if indices is None:
            indices = seed_target_indices(
                seeds=np.arange(len(self.signal.ch_names)),
                targets=np.arange(len(self.signal.ch_names)),
            )
        else:
            for group, channels in indices.items():
                print("jeff")
                # CHECK IF CHANNELS IS STR, IN WHICH CASE CHECK IF CHANNELS BEGINS WITH TYPE AND TAKE THE CHANNELS OF THAT TYPE, ELSE USE THE CHANNELS AS-IS

        self.indices = indices

    def _sort_processing_inputs(
        self,
        method: str,
        mode: str,
        indices: Union[dict[Union[str, list[str]]], None],
        fmin: Union[float, tuple, None],
        fmax: Union[float, tuple, None],
        fskip: int,
        faverage: bool,
        tmin: Union[float, None],
        tmax: Union[float, None],
        mt_bandwidth: Union[float, None],
        mt_adaptive: bool,
        mt_low_bias: bool,
        cwt_freqs: Union[list, np.array, None],
        cwt_n_cycles: Union[
            int, float, np.array[Union[int, float]], list[Union[int, float]]
        ],
        shuffle_group: Union[str, None],
        n_shuffles: Union[int, None],
        block_size: int,
        n_jobs: int,
    ) -> None:
        """"""

        self._sort_shuffled(shuffle_group=shuffle_group, n_shuffles=n_shuffles)
        self._sort_indices(indices=indices)

        self._method = method
        self._mode = mode
        self._fmin = fmin
        self._fmax = fmax
        self._fskip = fskip
        self._faverage = faverage
        self._tmin = tmin
        self._tmax = (tmax,)
        self._mt_bandwidth = mt_bandwidth
        self._mt_adaptive = mt_adaptive
        self._mt_low_bias = mt_low_bias
        self._cwt_freqs = cwt_freqs
        self._cwt_n_cycles = cwt_n_cycles
        self._block_size = block_size
        self._n_jobs = n_jobs

    def process(
        self,
        method: str,
        mode: str,
        indices: Optional[dict[Union[str, list[str]]]] = None,
        fmin: Optional[Union[float, tuple]] = None,
        fmax: Optional[Union[float, tuple]] = None,
        fskip: int = 0,
        faverage: bool = False,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        mt_bandwidth: Optional[float] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Union[list, np.array] = None,
        cwt_n_cycles: Union[
            int, float, np.array[Union[int, float]], list[Union[int, float]]
        ] = 7,
        shuffle_group: Optional[str] = None,
        n_shuffles: Optional[int] = None,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Test"""

        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self._sort_processing_inputs(
            method=method,
            mode=mode,
            indices=indices,
            fmin=fmin,
            fmax=fmax,
            fskip=fskip,
            faverage=faverage,
            tmin=tmin,
            tmax=tmax,
            mt_bandwidth=mt_bandwidth,
            mt_adaptive=mt_adaptive,
            mt_low_bias=mt_low_bias,
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles,
            shuffle_group=shuffle_group,
            n_shuffles=n_shuffles,
            block_size=block_size,
            n_jobs=n_jobs,
        )

        self._processed = True
