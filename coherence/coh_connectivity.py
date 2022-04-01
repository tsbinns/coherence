"""Classes for calculating connectivity between signals.

METHODS
-------
ConnectivityCoherence : subclass of the abstract base class 'ProcMethod'
-   Calculates the coherence (standard or imaginary) between signals.
"""

from typing import Optional, Union
from mne import concatenate_epochs
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs
from numpy.typing import NDArray
import numpy as np
from sklearn.utils import shuffle
from coh_exceptions import (
    InputTypeError,
    ProcessingOrderError,
)
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

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)

        # Initialises inputs of the object.
        self._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self.coherence = None
        self.coherence_dims = None
        self._method = None
        self._seeds = None
        self._targets = None
        self._indices = None
        self._mode = None
        self._fmin = None
        self._fmax = None
        self._fskip = None
        self._faverage = None
        self._tmin = None
        self._tmax = None
        self._mt_bandwidth = None
        self._mt_adaptive = None
        self._mt_low_bias = None
        self._cwt_freqs = None
        self._cwt_n_cycles = None
        self._shuffle_group = None
        self._n_shuffles = None
        self._shuffle_rng_seed = None
        self._block_size = None
        self._n_jobs = None

        # Initialises aspects of the object that indicate which methods have
        # been called (starting as 'False'), which can later be updated.
        self._shuffled = False

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
        if self.signal.data_dimensions not in supported_data_dims:
            raise InputTypeError(
                "Error when performing coherence analysis on the data:\nThe "
                f"preprocessed data is in the form {self.signal.power_dims}, "
                f"but only data in the form {supported_data_dims} is supported."
            )

        super()._sort_inputs()

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""

        self._indices = seed_target_indices(
            seeds=[
                i
                for i, name in enumerate(self.signal.data.ch_names)
                if name in self._seeds
            ],
            targets=[
                i
                for i, name in enumerate(self.signal.data.ch_names)
                if name in self._targets
            ],
        )

    def _sort_indices(self) -> None:
        """Sorts the inputs for generating MNE-readable indices for calculating
        connectivity between signals."""

        groups = ["_seeds", "_targets"]
        channel_types = self.signal.data.get_channel_types()
        for group in groups:
            channels = getattr(self, group)
            if channels is None:
                setattr(self, group, self.signal.data.ch_names)
            elif isinstance(channels, str):
                if channels[:5] == "type_":
                    setattr(
                        self,
                        group,
                        [
                            name
                            for i, name in enumerate(self.signal.data.ch_names)
                            if channel_types[i] == channels[5:]
                        ],
                    )
                else:
                    setattr(self, group, [channels])

        self._generate_indices()

    def _generate_shuffled_data(self) -> None:
        """Generates time-series data for a set of channels in which the order
        of epochs is randomly organised (i.e. 'shuffled') and adds it to the
        data."""

        shuffle_channels = getattr(self, f"_{self._shuffle_group}")
        data_to_shuffle = self.signal.data.copy()
        data_to_shuffle = data_to_shuffle.pick_channels(
            ch_names=shuffle_channels, ordered=True
        )
        if self._shuffle_rng_seed is not None:
            np.random.seed(self._shuffle_rng_seed)
        shuffled_data = []
        shuffled_channels = {name: [] for name in shuffle_channels}
        for shuffle_n in range(self._n_shuffles):
            epoch_order = np.arange(len(data_to_shuffle.events))
            np.random.shuffle(epoch_order)
            shuffled_data.append(
                concatenate_epochs(
                    [data_to_shuffle[epoch_order]], verbose=False
                )
            )
            shuffled_data[shuffle_n].rename_channels(
                {
                    name: f"SHUFFLED[{shuffle_n}]_{name}"
                    for name in shuffle_channels
                }
            )
            [
                shuffled_channels[name].append(f"SHUFFLED[{shuffle_n}]_{name}")
                for name in shuffle_channels
            ]

        self.signal.data.add_channels(shuffled_data)
        self._shuffled_channels = shuffled_channels

        if self._verbose:
            print(
                "Creating epoch-shuffled data for the following channels:\n"
                f"{shuffle_channels}"
            )

    def _add_shuffled_to_indices(self) -> None:
        """Adds newly-created epoch-shuffled channels of data to the
        connectivity indices."""

        shuffled_channel_names = []
        [
            shuffled_channel_names.extend(value)
            for value in self._shuffled_channels.values()
        ]
        if self._shuffle_group == "seeds":
            self._seeds.extend(shuffled_channel_names)
        else:
            self._targets.extend(shuffled_channel_names)

        self._generate_indices()

    def _sort_shuffles(self) -> None:
        """Organises the shuffling of time-series data for a set of channels in
        which the order of epochs is randomly organised (i.e. 'shuffled'),
        adding these new channels to the data and updating the connectivity
        indices.
        """

        self._generate_shuffled_data()
        self._add_shuffled_to_indices()

        self._shuffled = True

    def _sort_processing_inputs(self) -> None:
        """Converts the connectivity seeds and targets into channel indices for
        the connectivity analysis, and generates epoch-shuffled data, if
        requested."""

        self._sort_indices()

        if self._cwt_freqs is not None:
            self._cwt_freqs = np.arange(
                start=self._cwt_freqs[0], stop=self._cwt_freqs[1] + 1
            )

        if self._shuffle_group is not None:
            self._sort_shuffles()

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""

        connectivity = spectral_connectivity_epochs(
            data=self.signal.data,
            method=self._method,
            indices=self._indices,
            sfreq=self.signal.data.info["sfreq"],
            mode=self._mode,
            fmin=self._fmin,
            fmax=self._fmax,
            fskip=self._fskip,
            faverage=self._faverage,
            tmin=self._tmin,
            tmax=self._tmax,
            mt_bandwidth=self._mt_bandwidth,
            mt_adaptive=self._mt_adaptive,
            mt_low_bias=self._mt_low_bias,
            cwt_freqs=self._cwt_freqs,
            cwt_n_cycles=self._cwt_n_cycles,
            block_size=self._block_size,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        )

        if self._shuffled:
            self._sort_shuffled_data(results=connectivity)

        self._sort_dimensions()

    def process(
        self,
        method: str,
        mode: str,
        seeds: Optional[Union[str, list[str]]] = None,
        targets: Optional[Union[str, list[str]]] = None,
        fmin: Optional[Union[float, tuple]] = None,
        fmax: Optional[Union[float, tuple]] = float("inf"),
        fskip: int = 0,
        faverage: bool = False,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        mt_bandwidth: Optional[float] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Optional[Union[list, NDArray]] = None,
        cwt_n_cycles: Union[int, float, NDArray] = 7,
        shuffle_group: Optional[str] = None,
        n_shuffles: Optional[int] = None,
        shuffle_rng_seed: Optional[int] = None,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        spectral_connectivity_epochs function of the mne-connectivity package.

        PARAMETERS
        ----------
        method : str
        -   The method for calculating connectivity.
        -   Supported inputs are: 'coh' - standard coherence; 'cohy' -
            coherency; 'imcoh' - imaginary part of coherence; 'plv' -
            phase-locking value; 'ciplv' - corrected imaginary phase-locking
            value; 'ppc' - pairwise phase consistency; 'pli' - phase lag index;
            'pli2_unbiased' - unbiased estimator of squared phase lag index;
            'wpli' - weighted phase lag index; 'wpli2_debiased' - debiased
            estimator of squared weighted phase lag index.

        mode : str
        -   The mode for calculating connectivity using 'method'.
        -   Supported inputs are: 'multitaper'; 'fourier'; and 'cwt_morlet'.

        seeds : str | list[str] | None; default None
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.
        -   If None, all channels will be used as seeds.

        targets : str | list[str] | None; default None
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.
        -   If None, all channels will be used as targets.

        fmin : float | tuple | None
        -   The lower frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            lower frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their lower frequencies.
        -   If None, no lower frequency is used.

        fmax : float | tuple; default infinite
        -   The higher frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            higher frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their higher frequencies.
        -   If an infinite float, no higher frequency is used.

        fskip : int; default 0
        -   Omit every 'fskip'+1th frequency bin to decimate the frequency
            domain.
        -   If 0, no bins are skipped.

        faverage : bool; default False
        -   Whether or not to average the connectivity values for each frequency
            band.

        tmin : float | None; default None
        -   Time to start the connectivity estimation.
        -   If None, the data is used from the beginning.

        tmax : float | None; default None
        -   Time to end the connectivity estimation.
        -   If None, the data is used until the end.

        mt_bandwidth : float | None
        -   The bandwidth, in Hz, of the multitaper windowing function.
        -   Only used if 'mode' is 'multitaper'.

        mt_adaptive : bool; default False
        -   Whether or not to use adaptive weights to comine the tapered spectra
            into power spectra.
        -   Only used if 'mode' is 'multitaper'.

        mt_low_bias : bool: default True
        -   Whether or not to only use tapers with > 90% spectral concentration
            within bandwidth.
        -   Only used if 'mode' is 'multitaper'.

        cwt_freqs: list[int | float] | array[int | float] | None
        -   The frequencies of interest to calculate connectivity for.
        -   Only used if 'mode' is 'cwt_morlet'. In this case, 'cwt_freqs'
            cannot be None.

        cwt_n_cycles: int | float | array[int | float]; default 7
        -   The number of cycles to use when calculating connectivity.
        -   If an single integer or float, this number of cycles is for each
            frequency.
        -   If an array, the entries correspond to the number of cycles to use
            for each frequency being analysed.
        -   Only used if 'mode' is 'cwt_morlet'.

        shuffle_group: str | None
        -   The group of channels to create epoch-shuffled data for. Supported
            groups are 'seeds' or 'targets', corresponding to the channels in
            those groups.
        -   If None, no shuffled data is created or analysed.

        n_shuffles : int | None
        -   How many times to create epoch-shuffled data for each channel in
            'shuffle_group', whose connectivity results are then averaged over.
        -   Only used if 'shuffle_group' is not None. In this case, 'n_shuffles'
            cannot be None.

        shuffle_rng_sheed : int | None
        -   The seed to use for the random number generator for shuffling the
            epoch order.
        -   If None, no seed is used.
        -   Only used if 'shuffle_group' is not None.

        block_size : int; default 1000
        -   The number of connections to compute at once.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.
        """

        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self._method = method
        self._mode = mode
        self._seeds = seeds
        self._targets = targets
        self._fmin = fmin
        self._fmax = fmax
        self._fskip = fskip
        self._faverage = faverage
        self._tmin = tmin
        self._tmax = tmax
        self._mt_bandwidth = mt_bandwidth
        self._mt_adaptive = mt_adaptive
        self._mt_low_bias = mt_low_bias
        self._cwt_freqs = cwt_freqs
        self._cwt_n_cycles = cwt_n_cycles
        self._shuffle_group = shuffle_group
        self._n_shuffles = n_shuffles
        self._shuffle_rng_seed = shuffle_rng_seed
        self._block_size = block_size
        self._n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

        self._processed = True
