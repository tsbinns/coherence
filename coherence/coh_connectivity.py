"""Classes for calculating connectivity between signals.

CLASSES
-------
ConnectivityCoherence : subclass of the abstract base class 'ProcMethod'
-   Calculates the coherence (standard or imaginary) between signals.

ConectivityMultivariate : subclass of the abstract base class 'ProcMethod'
-   Calculates the multivariate connectivity (multivariate interaction measure,
    MIM, or maximised imaginary coherence, MIC) between signals.
"""

from copy import deepcopy
from typing import Optional, Union
from mne import Epochs
from mne_connectivity import (
    seed_target_indices,
    spectral_connectivity_epochs,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
)
from numpy.typing import NDArray
import numpy as np
from coh_connectivity_computations import multivariate_connectivity
from coh_exceptions import (
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_handle_entries import (
    combine_vals_list,
    ordered_list_from_dict,
    rearrange_axes,
    separate_vals_string,
    unique,
)
from coh_processing_methods import ProcMethod
from coh_progress_bar import ProgressBar
import coh_signal
from coh_saving import save_object, save_dict


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
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.

    get_results
    -   Extracts and returns results.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)

        # Initialises inputs of the object.
        self._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
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
        self._average_windows = None
        self._average_timepoints = None
        self._block_size = None
        self._n_jobs = None

        # Initialises aspects of the object that indicate which methods have
        # been called (starting as 'False'), which can later be updated.
        self._timepoints_averaged = False

    def _sort_inputs(self) -> None:
        """Checks the inputs to the object to ensure that they match the
        requirements for processing and assigns inputs.

        RAISES
        ------
        ValueError
        -   Raised if the dimensions of the data in the Signal object is not
            supported.
        """

        supported_data_dims = ["windows", "epochs", "channels", "timepoints"]
        if self.signal.data_dimensions != supported_data_dims:
            raise ValueError(
                "Error when trying to perform coherence analysis on the "
                "data:\nData in the Signal object has the dimensions "
                f"{self.signal.data_dimensions}, but only data with dimensions "
                f"{supported_data_dims} is supported."
            )

        super()._sort_inputs()

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""

        self._indices = seed_target_indices(
            seeds=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self._seeds
            ],
            targets=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self._targets
            ],
        )

    def _sort_indices(self) -> None:
        """Sorts the inputs for generating MNE-readable indices for calculating
        connectivity between signals."""

        groups = ["_seeds", "_targets"]
        ch_types = self.signal.data[0].get_channel_types()
        for group in groups:
            channels = getattr(self, group)
            if channels is None:
                channels = deepcopy(self.signal.data[0].ch_names)
            elif isinstance(channels, str):
                if channels[:5] == "type_":
                    desired_type = channels[5:]
                    channels = [
                        name
                        for i, name in enumerate(self.signal.data[0].ch_names)
                        if ch_types[i] == desired_type
                    ]
                else:
                    channels = [channels]
            setattr(self, group, channels)

        self._generate_indices()

    def _sort_processing_inputs(self) -> None:
        """Converts the connectivity seeds and targets into channel indices for
        the connectivity analysis, and generates epoch-shuffled data, if
        requested."""

        self._sort_indices()

        if self._cwt_freqs is not None:
            self._cwt_freqs = np.arange(
                start=self._cwt_freqs[0], stop=self._cwt_freqs[1] + 1
            )

    def _average_windows_results(self) -> None:
        """Averages the connectivity results across windows.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the windows have already been averaged across.
        """

        if self._windows_averaged:
            raise ProcessingOrderError(
                "Error when averaging the connectivity results across "
                "windows:\nResults have already been averaged across windows."
            )

        n_windows = len(self.results)
        coherence = []
        for results in self.results:
            coherence.append(results.get_data())
        coherence = np.asarray(coherence).mean(axis=0)
        self.results = [
            SpectroTemporalConnectivity(
                data=coherence,
                freqs=self.results[0].freqs,
                times=self.results[0].times,
                n_nodes=self.results[0].n_nodes,
                names=self.results[0].names,
                indices=self.results[0].indices,
                method=self.results[0].method,
                n_epochs_used=self.results[0].n_epochs_used,
            )
        ]

        self._windows_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

    def _average_timepoints_results(self) -> None:
        """Averages the connectivity results across timepoints.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the timepoints have already been averaged across.

        UnavailableProcessingError
        -   Raised if timepoints are not present in the results.
        """

        if self._timepoints_averaged:
            raise ProcessingOrderError(
                "Error when processing the connectivity results: Trying to "
                "average the data across timepoints, however this has already "
                "been performed."
            )

        if "timepoints" not in self.results_dims:
            raise UnavailableProcessingError(
                "Error when attempting to average the timepoints in the "
                "connectivity results:\n There is no timepoints axis present "
                f"in the data. The present axes are: \n{self.results_dims}"
            )

        timepoints_i = self.results_dims.index("timepoints")
        n_timepoints = np.shape(self.results[0].get_data())[timepoints_i]
        for i, results in enumerate(self.results):
            self.results[i] = SpectralConnectivity(
                data=np.mean(results.get_data(), axis=timepoints_i),
                freqs=self.results[0].freqs,
                n_nodes=self.results[0].n_nodes,
                names=self.results[0].names,
                indices=self.results[0].indices,
                method=self.results[0].method,
                n_epochs_used=self.results[0].n_epochs_used,
            )
        self._results_dims.pop(timepoints_i + 1)

        self._timepoints_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_timepoints} timepoints.\n")

    def _establish_coherence_dimensions(self) -> None:
        """Establishes the dimensions of the coherence results.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if dimensions for the results are not listed for the analysis
            method used to calculate the results.
        """

        supported_modes = ["multitaper", "fourier", "cwt_morlet"]

        self._results_dims = ["windows"]
        if self._mode in ["multitaper", "fourier"]:
            self._results_dims.extend(["channels", "frequencies"])
        elif self._mode in ["cwt_morlet"]:
            self._results_dims.extend(["channels", "frequencies", "timepoints"])
        else:
            raise UnavailableProcessingError(
                "Error when sorting the results of the connectivity analysis:\n"
                f"The analysis mode '{self._mode}' does not have an associated "
                "dimension for the results axes.\nOnly methods "
                f"'{supported_modes}' are supported."
            )

    def _sort_dimensions(self) -> None:
        """Establishes dimensions of the coherence results and averages across
        windows and/or timepoints, if requested."""

        self._establish_coherence_dimensions()

        if self._average_windows:
            self._average_windows_results()

        if self._average_timepoints:
            self._average_timepoints_results()

    def _generate_node_ch_names(self) -> list[list[str]]:
        """Converts the indices of channels in the connectivity results to their
        channel names.

        RETURNS
        -------
        node_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_names = [[], []]

        for group_i, indices in enumerate(self.results[0].indices):
            for index in indices:
                node_names[group_i].append(self.results[0].names[index])

        return node_names

    def _generate_node_ch_types(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the types of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_types : list[list[str]]
        -   List containing two sublists consisting of the channel types of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_ch_types = [[], []]
        ch_types = {}
        for name in self.results[0].names:
            ch_types[name] = self.signal.data[0].get_channel_types(picks=name)[
                0
            ]

        for group_i in range(2):
            node_ch_types[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i], dict_to_order=ch_types
            )

        return node_ch_types

    def _generate_node_ch_reref_types(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the rereferencing types of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_reref_types : list[list[str]]
        -   List containing two sublists consisting of the channel types of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_reref_types = [[], []]

        for group_i in range(2):
            node_reref_types[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i],
                dict_to_order=self.extra_info["ch_reref_types"],
            )

        return node_reref_types

    def _generate_node_ch_coords(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the coordinates of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_coords : list[list[str]]
        -   List containing two sublists consisting of the channel coordinates
            of the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_coords = [[], []]
        ch_coords = {
            name: self.signal.get_coordinates(name)[0]
            for name in self.results[0].names
        }

        for group_i in range(2):
            node_ch_coords[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i], dict_to_order=ch_coords
            )

        return node_ch_coords

    def _generate_node_ch_regions(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the regions of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_regions : list[list[str]]
        -   List containing two sublists consisting of the channel regions of
            the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_regions = [[], []]

        for group_i in range(2):
            node_ch_regions[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i],
                dict_to_order=self.extra_info["ch_regions"],
            )

        return node_ch_regions

    def _generate_node_ch_hemispheres(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the hemispheres of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_hemispheres : list[list[str]]
        -   List containing two sublists consisting of the hemispheres of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_ch_hemispheres = [[], []]

        for group_i in range(2):
            node_ch_hemispheres[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i],
                dict_to_order=self.extra_info["ch_hemispheres"],
            )

        return node_ch_hemispheres

    def _generate_node_lateralisation(
        self, node_ch_hemispheres: list[list[str]]
    ) -> list[str]:
        """Gets the lateralisation of the channels in the connectivity node.
        -   Can either be "contralateral" if the seed and target are from
            different hemispheres, or "ipsilateral" if the seed and target are
            from the same hemisphere.

        PARAMETERS
        ----------
        node_ch_hemispheres : list[list[str]]
        -   Hemispheres of the seed and target channels of each connectivity
            node.
        -   Indication of the hemispheres should be binary in nature, e.g. "L"
            and "R", or "Left" and "Right", not "L" and "Left" and "R" and
            "Right".

        RETURNS
        -------
        node_lateralisation : list[str]
        -   Lateralisation ("contralateral" or "ipsilateral") of each
            connectivity node.
        """

        node_lateralisation = []

        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                node_lateralisation.append("contralateral")
            else:
                node_lateralisation.append("ipsilateral")

        return node_lateralisation

    def _generate_node_ch_epoch_orders(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the epoch orders of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_epoch_orders : list[str]
        -   List containing the epoch orders of the seeds and targets for each
            node in the connectivity results. If either the seed or target has
            a 'shuffled' epoch order, the epoch order of the node is 'shuffled',
            otherwise it is 'original'.
        """

        seed_epoch_orders = ordered_list_from_dict(
            list_order=node_ch_names[0],
            dict_to_order=self.extra_info["ch_epoch_orders"],
        )
        target_epoch_orders = ordered_list_from_dict(
            list_order=node_ch_names[1],
            dict_to_order=self.extra_info["ch_epoch_orders"],
        )

        node_epoch_orders = []
        for seed_order, target_order in zip(
            seed_epoch_orders, target_epoch_orders
        ):
            if seed_order == "original" and target_order == "original":
                node_epoch_orders.append("original")
            else:
                node_epoch_orders.append("shuffled")

        return node_epoch_orders

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""

        self.extra_info["node_ch_names"] = self._generate_node_ch_names()
        self.extra_info["node_ch_types"] = self._generate_node_ch_types(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info[
            "node_ch_reref_types"
        ] = self._generate_node_ch_reref_types(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info["node_ch_coords"] = self._generate_node_ch_coords(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info["node_ch_regions"] = self._generate_node_ch_regions(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info[
            "node_ch_hemispheres"
        ] = self._generate_node_ch_hemispheres(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info[
            "node_lateralisation"
        ] = self._generate_node_lateralisation(
            node_ch_hemispheres=self.extra_info["node_ch_hemispheres"]
        )
        self.extra_info[
            "node_ch_epoch_orders"
        ] = self._generate_node_ch_epoch_orders(
            node_ch_names=self.extra_info["node_ch_names"]
        )

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""

        connectivity = []
        for i, data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"Computing connectivity for window {i+1} of "
                    f"{len(self.signal.data)}.\n"
                )
            connectivity.append(
                spectral_connectivity_epochs(
                    data=data,
                    method=self._method,
                    indices=self._indices,
                    sfreq=data.info["sfreq"],
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
            )
            if self._method == "imcoh":
                connectivity[i] = SpectroTemporalConnectivity(
                    data=np.abs(connectivity[i].get_data()),
                    freqs=connectivity[i].freqs,
                    times=connectivity[i].times,
                    n_nodes=connectivity[i].n_nodes,
                    names=connectivity[i].names,
                    indices=connectivity[i].indices,
                    method=connectivity[i].method,
                    n_epochs_used=connectivity[i].n_epochs_used,
                )
        self.results = connectivity

        self._sort_dimensions()
        self._generate_extra_info()

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
        average_windows: bool = False,
        average_timepoints: bool = False,
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

        average_windows : bool; default False
        -   Whether or not to average connectivity results across windows.

        average_timepoints : bool; default False
        -   Whether or not to average connectivity results across timepoints.

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
        self._average_windows = average_windows
        self._average_timepoints = average_timepoints
        self._block_size = block_size
        self._n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

        self._processed = True
        self.processing_steps["connectivity_coherence"] = {
            "method": method,
            "mode": mode,
            "seeds": seeds,
            "targets": targets,
            "fmin": fmin,
            "fmax": fmax,
            "fskip": fskip,
            "faverage": faverage,
            "tmin": tmin,
            "tmax": tmax,
            "mt_bandwidth": mt_bandwidth,
            "mt_adaptive": mt_adaptive,
            "mt_low_bias": mt_low_bias,
            "cwt_freqs": cwt_freqs,
            "cwt_n_cycles": cwt_n_cycles,
            "average_windows": average_windows,
            "average_timepoints": average_timepoints,
        }

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   Whether or not the user is asked to confirm to overwrite a
            pre-existing file if one exists.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_object(
            to_save=self,
            fpath=fpath,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def save_results(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the results and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved.

        ftype : str | None; default None
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_dict(
            to_save=self.results_as_dict(),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def results_as_dict(self) -> dict:
        """Returns the results and additional information as a dictionary.

        RETURNS
        -------
        dict
        -   The results and additional information stored as a dictionary.
        """

        dimensions = self._get_optimal_dims()
        results = self.get_results(dimensions=dimensions)

        return {
            f"connectivity-{self._method}": results.tolist(),
            f"connectivity-{self._method}_dimensions": dimensions,
            "freqs": self.results[0].freqs,
            "seed_names": self.extra_info["node_ch_names"][0],
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self.extra_info["node_ch_names"][1],
            "target_types": self.extra_info["node_ch_types"][1],
            "target_coords": self.extra_info["node_ch_coords"][1],
            "target_regions": self.extra_info["node_ch_regions"][1],
            "target_hemispheres": self.extra_info["node_ch_hemispheres"][1],
            "target_reref_types": self.extra_info["node_ch_reref_types"][1],
            "node_lateralisation": self.extra_info["node_lateralisation"],
            "node_epoch_orders": self.extra_info["node_ch_epoch_orders"],
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }

    def get_results(self, dimensions: Union[list[str], None] = None) -> NDArray:
        """Extracts and returns results.

        PARAMETERS
        ----------
        dimensions : list[str] | None;  default None
        -   The dimensions of the results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        results : numpy array
        -   The results.
        """

        if dimensions is None:
            dimensions = self.results_dims

        if self._windows_averaged:
            results = self.results[0].get_data()
        else:
            results = []
            for mne_obj in self.results:
                results.append(mne_obj.get_data())
            results = np.asarray(results)

        results = rearrange_axes(
            obj=results, old_order=self.results_dims, new_order=dimensions
        )

        return deepcopy(results)


class ConnectivityMultivariate(ProcMethod):
    """Calculates the multivariate connectivity (multivariate interaction
    measure, MIM, or maximised imaginary coherence, MIC) between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs multivariate connectivity analysis.

    save_object
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.

    get_results
    -   Extracts and returns results.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)

        # Initialises inputs of the object.
        self._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
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
        self._average_windows = None
        self._block_size = None
        self._n_jobs = None
        self._separated_names = None

    def _sort_inputs(self) -> None:
        """Checks the inputs to the object to ensure that they match the
        requirements for processing and assigns inputs.

        RAISES
        ------
        ValueError
        -   Raised if the dimensions of the data in the Signal object is not
            supported.
        """

        supported_data_dims = ["windows", "epochs", "channels", "timepoints"]
        if self.signal.data_dimensions != supported_data_dims:
            raise ValueError(
                "Error when trying to perform coherence analysis on the "
                "data:\nData in the Signal object has the dimensions "
                f"{self.signal.data_dimensions}, but only data with dimensions "
                f"{supported_data_dims} is supported."
            )

        super()._sort_inputs()

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
        average_windows: bool = False,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        spectral_connectivity_epochs function of the mne-connectivity package to
        generate coherency values for the computation of multivariate connectivity
        metrics.

        PARAMETERS
        ----------
        method : str
        -   The multivariate connectivity metric to compute.
        -   Supported inputs are: 'mim' - multivariate interaction measure; and
            'mic' - maximised imaginary coherence.

        mode : str
        -   The mode for calculating coherency.
        -   Supported inputs are: 'multitaper'; 'fourier'; and 'cwt_morlet'.

        seeds : str | list[str] | None; default None
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'. In this case, channels belonging
            to each type with different epoch orders and rereferencing types will be
            handled separately.
        -   If a list of strings, each entry of the list should be a channel
            name.
        -   If None, all channels will be used as seeds.

        targets : str | list[str] | None; default None
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'. In this case, channels belonging
            to each type with different epoch orders and rereferencing types will be
            handled separately.
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

        average_windows : bool; default False
        -   Whether or not to average connectivity results across windows.

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
        self._average_windows = average_windows
        self._block_size = block_size
        self._n_jobs = n_jobs

        self._sort_processing_inputs()

        if self._verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data) * len(self._indices) * 2,
                title="Computing connectivity",
            )

        self._get_results()

        if self._verbose:
            self._progress_bar.close()

        self._processed = True
        self.processing_steps["connectivity_multivariate"] = {
            "method": method,
            "mode": mode,
            "seeds": seeds,
            "targets": targets,
            "fmin": fmin,
            "fmax": fmax,
            "fskip": fskip,
            "faverage": faverage,
            "tmin": tmin,
            "tmax": tmax,
            "mt_bandwidth": mt_bandwidth,
            "mt_adaptive": mt_adaptive,
            "mt_low_bias": mt_low_bias,
            "cwt_freqs": cwt_freqs,
            "cwt_n_cycles": cwt_n_cycles,
            "average_windows": average_windows,
        }

    def _sort_processing_inputs(self) -> None:
        """Converts the connectivity seeds and targets into channel indices for
        the connectivity analysis, and generates epoch-shuffled data, if
        requested."""

        self._sort_indices()

        if self._cwt_freqs is not None:
            self._cwt_freqs = np.arange(
                start=self._cwt_freqs[0], stop=self._cwt_freqs[1] + 1
            )

    def _sort_indices(self) -> None:
        """Sorts the inputs for generating MNE-readable indices for calculating
        connectivity between signals."""

        groups = ["_seeds", "_targets"]
        expand_groups = False
        for group in groups:
            channels = getattr(self, group)
            if channels is None:
                channels = deepcopy(self.signal.data[0].ch_names)
            elif isinstance(channels, str):
                if channels[:5] == "type_":
                    expand_groups = True
                    channels = self._seeds_targets_from_type(
                        ch_type=channels[5:]
                    )
                else:
                    channels = [channels]
            setattr(self, group, channels)

        if expand_groups:
            self._expand_seeds_targets()

        self._generate_indices()

    def _seeds_targets_from_type(self, ch_type: str) -> list[str]:
        """Gets channel names to use for connectivity seeds or targets for a
        particular channel type, grouping channels based on their rereferencing
        and epoch order types.

        PARAMETERS
        ----------
        ch_type : str
        -   The channel type to get seed/target names for.

        RETURNS
        -------
        channels : list[str]
        -   Names of channels grouped according to their rereferencing and epoch
            order types.
        """

        channels = []
        chs_of_type = []
        ch_types = self.signal.data[0].get_channel_types()
        for ch_i, ch_name in enumerate(self.signal.data[0].ch_names):
            if ch_types[ch_i] == ch_type:
                chs_of_type.append(ch_name)

        ch_reref_types = []
        ch_epoch_orders = []
        ch_hemispheres = []
        for ch_name in chs_of_type:
            ch_reref_types.append(self.extra_info["ch_reref_types"][ch_name])
            ch_epoch_orders.append(self.extra_info["ch_epoch_orders"][ch_name])
            ch_hemispheres.append(self.extra_info["ch_hemispheres"][ch_name])
        reref_types = unique(ch_reref_types)
        epoch_orders = unique(ch_epoch_orders)
        hemispheres = unique(ch_hemispheres)

        for reref_type in reref_types:
            for epoch_order in epoch_orders:
                for hemisphere in hemispheres:
                    channels.append([])
                    for ch_i, ch_name in enumerate(chs_of_type):
                        if (
                            ch_reref_types[ch_i] == reref_type
                            and ch_epoch_orders[ch_i] == epoch_order
                            and ch_hemispheres[ch_i] == hemisphere
                        ):
                            channels[-1].append(ch_name)

        return channels

    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group.
        -   Should be used when seeds and/or targets have been automatically
            generated based on channel types."""

        seeds = []
        targets = []
        for seed in self._seeds:
            for target in self._targets:
                seeds.append(seed)
                targets.append(target)

        self._seeds = seeds
        self._targets = targets

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""

        self._indices = []
        for seeds, targets in zip(self._seeds, self._targets):
            ch_names = [*seeds, *targets]
            ch_idcs = [
                ch_i
                for ch_i, name in enumerate(self.signal.data[0].ch_names)
                if name in ch_names
            ]
            self._indices.append(
                seed_target_indices(seeds=ch_idcs, targets=ch_idcs)
            )

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""

        connectivity = []
        for win_i, win_data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"Computing connectivity for window {win_i+1} of "
                    f"{len(self.signal.data)}.\n"
                )
            coherency = self._get_cohy(win_data)
            connectivity.append(self._get_multivariate_results(coherency))
        self.results = connectivity

        self._sort_dimensions()
        self._generate_extra_info()

    def _get_cohy(self, data: Epochs) -> list[SpectralConnectivity]:
        """For the data of a single window, calculates the coherency between all
        seeds and targets for use in computing multivariate measures using the
        implementation of MNE's 'spectral_connectivity_epochs'.
        -   Any resulting temporal data is averaged over to give a single
            connectivity value per seed-target pair.

        PARAMETERS
        ----------
        data : MNE Epochs
        -   The data for a single window.

        RETURNS
        -------
        coherency : list[MNE SpectralConnectivity]
        -   List containing the coherency for each indices group.
        """

        coherency = []
        for con_i, indices in enumerate(self._indices):
            if self._verbose:
                print(
                    f"Computing coherency for seed-target group {con_i} of "
                    f"{len(self._indices)}\n."
                )
            results = spectral_connectivity_epochs(
                data=data,
                method="cohy",
                indices=indices,
                sfreq=data.info["sfreq"],
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
            if isinstance(results, SpectroTemporalConnectivity):
                results = SpectralConnectivity(
                    data=np.mean(results.get_data(), axis=-1),
                    freqs=results.freqs,
                    n_nodes=results.n_nodes,
                    names=results.names,
                    indices=results.indices,
                    method=results.method,
                    n_epochs_used=results.n_epochs_used,
                )
            coherency.append(results)
            if self._verbose:
                self._progress_bar.update_progress()

        return coherency

    def _get_multivariate_results(
        self, data: list[SpectralConnectivity]
    ) -> None:
        """For a single window, computes the multivariate connectivity results
        using the coherency data.

        PARAMETERS
        ----------
        data : list[MNE SpectralConnectivity]
        -   The coherency data for a single window. Each entry should contain
            the connectivity values for all seed-seed, seed-target, and
            target-target pairs within a single seed-target group.

        RETURNS
        -------
        MNE SpectralConnectivity
        -   The multivariate connectivity results of a single window for all
            seed-target pairs.
        """

        connectivity = []
        for con_i, coherency in enumerate(data):
            if self._verbose:
                print(
                    f"Computing '{self._method}' for seed-target group {con_i} "
                    f"of {len(self._indices)}\n."
                )
            cohy_matrix = self._get_cohy_matrix(coherency)
            results = multivariate_connectivity(
                data=cohy_matrix,
                method=self._method,
                n_group_a=len(self._seeds[con_i]),
                n_group_b=len(self._targets[con_i]),
            )
            connectivity.append(results)
            if self._verbose:
                self._progress_bar.update_progress()

        return self._multivariate_to_mne(
            data=np.asarray(connectivity),
            freqs=data[0].freqs,
            n_epochs_used=data[0].n_epochs_used,
        )

    def _get_cohy_matrix(self, data: SpectralConnectivity) -> NDArray:
        """Converts the coherency data with dimensions [connections x
        frequencies] - where the number of connections is equal to the number of
        nodes squared - into a three-dimensional matrix with dimensions [nodes x
        nodes x frequencies].

        PARAMETERS
        ----------
        data : MNE SpectralConnectivity
        -   MNE connectivity object containing the coherency values for all
            possible seed-target pairs for two groups of signals.

        RETURNS
        -------
        data_matrix : numpy Array
        -   A three-dimensional array containing the coherency values for all
            possible connections between the seed-target pairs in the first two
            dimensions, and frequencies in the third dimension.
        """

        data_vals = data.get_data()
        n_nodes = len(np.unique(data.indices[0]))
        n_freqs = len(data.freqs)
        data_matrix = np.empty((n_nodes, n_nodes, n_freqs), dtype="complex128")
        for freq_i in range(n_freqs):
            data_matrix[:, :, freq_i] = np.reshape(
                data_vals[:, freq_i], (n_nodes, n_nodes)
            )

        return data_matrix

    def _multivariate_to_mne(
        self, data: NDArray, freqs: NDArray, n_epochs_used: int
    ) -> SpectralConnectivity:
        """Converts results of the multivariate connectivity analsyis stored as
        a numpy array into an MNE SpectralConnectivity object.

        PARAMETERS
        ----------
        data : numpy array
        -   Results of the multivariate connectivity analysis with dimensions
            [nodes x frequencies].

        freqs : numpy array
        -   The frequencies in the connectivity data.

        n_epochs_used : int
        -   The number of epochs used to compute connectivity.

        RETURNS
        -------
        MNE SpectralConnectivity
        -   Results converted to an appropriate MNE object.
        """

        names, indices = self._get_names_indices_mne()

        return SpectralConnectivity(
            data=data,
            freqs=freqs,
            n_nodes=len(self._seeds),
            names=names,
            indices=indices,
            method=self._method,
            n_epochs_used=n_epochs_used,
        )

    def _get_names_indices_mne(self) -> tuple[NDArray, NDArray]:
        """Gets the names and indices of seed and targets in the connectivity
        analysis for use in an MNE connectivity object.
        -   As MNE connectivity objects only support seed-target pair names and
            indices between two channels, the names of channels in each group of
            seeds and targets are combined together, and the indices derived
            from these combined names.

        RETURNS
        -------
        unique_names : numpy array
        -   Names of the channels combined for each group.

        indices : numpy array
        -   Array with two entries containing the seed and target indices,
            respectively, of the connectivity results based on the combined
            channel names in 'unique_names'.
        """

        seed_names = []
        target_names = []
        for seeds, targets in zip(self._seeds, self._targets):
            seed_names.append(combine_vals_list(seeds))
            target_names.append(combine_vals_list(targets))
        unique_names = [*unique(seed_names), *unique(target_names)]

        indices = [[], []]
        for seeds, targets in zip(self._seeds, self._targets):
            indices[0].append(unique_names.index(combine_vals_list(seeds)))
            indices[1].append(unique_names.index(combine_vals_list(targets)))

        return np.asarray(unique_names), np.asarray(indices)

    def _sort_dimensions(self) -> None:
        """Establishes dimensions of the connectivity results and averages
        across windows, if requested."""

        self._results_dims = ["windows", "channels", "frequencies"]

        if self._average_windows:
            self._average_windows_results()

    def _average_windows_results(self) -> None:
        """Averages the connectivity results across windows.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the windows have already been averaged across.
        """

        if self._windows_averaged:
            raise ProcessingOrderError(
                "Error when averaging the connectivity results across "
                "windows:\nResults have already been averaged across windows."
            )

        n_windows = len(self.results)
        connectivity = []
        for results in self.results:
            connectivity.append(results.get_data())
        connectivity = np.asarray(connectivity).mean(axis=0)
        self.results = [
            SpectralConnectivity(
                data=connectivity,
                freqs=self.results[0].freqs,
                n_nodes=self.results[0].n_nodes,
                names=self.results[0].names,
                indices=self.results[0].indices,
                method=self.results[0].method,
                n_epochs_used=self.results[0].n_epochs_used,
            )
        ]

        self._windows_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""

        self.extra_info["node_ch_names"] = self._generate_node_ch_names()
        self.extra_info["node_ch_types"] = self._generate_node_ch_types()
        self.extra_info[
            "node_ch_reref_types"
        ] = self._generate_node_ch_reref_types()
        self.extra_info["node_ch_coords"] = self._generate_node_ch_coords()
        self.extra_info["node_ch_regions"] = self._generate_node_ch_regions()
        (
            self.extra_info["node_ch_hemispheres"],
            node_single_hemispheres,
        ) = self._generate_node_ch_hemispheres()
        self.extra_info[
            "node_lateralisation"
        ] = self._generate_node_lateralisation(
            node_single_hemispheres=node_single_hemispheres,
        )
        self.extra_info[
            "node_ch_epoch_orders"
        ] = self._generate_node_ch_epoch_orders()

    def _generate_node_ch_names(self) -> list[list[str]]:
        """Converts the indices of channels in the connectivity results to their
        channel names.

        RETURNS
        -------
        node_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        self._separated_names = []
        for combined_names in self.results[0].names:
            self._separated_names.append(
                separate_vals_string(combined_names, " & ")
            )

        node_names = [[], []]
        for group_i, indices in enumerate(self.results[0].indices):
            for index in indices:
                node_names[group_i].append(self.results[0].names[index])

        return node_names

    def _generate_node_ch_types(self) -> list[list[str]]:
        """Gets the types of channels in the connectivity results.

        RETURNS
        -------
        node_ch_types : list[list[str]]
        -   List containing two sublists consisting of the channel types of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        -   If the types of each channel in a seed/target for a given node are
            identical, this type is given as a string, otherwise the unique
            types are taken and joined into a single string by the " & "
            characters.
        """

        ch_types = {}
        for node_i, combined_names in enumerate(self.results[0].names):
            ch_types[combined_names] = []
            for name in self._separated_names[node_i]:
                ch_types[combined_names].append(
                    self.signal.data[0].get_channel_types(picks=name)[0]
                )
            unique_types = unique(ch_types[combined_names])
            if len(unique_types) > 1:
                unique_types = [combine_vals_list(unique_types)]
            ch_types[combined_names] = unique_types[0]

        node_ch_types = [[], []]
        for group_i in range(2):
            for name in self.extra_info["node_ch_names"][group_i]:
                node_ch_types[group_i].append(ch_types[name])

        return node_ch_types

    def _generate_node_ch_reref_types(self) -> list[list[str]]:
        """Gets the rereferencing types of channels in the connectivity results.

        RETURNS
        -------
        node_reref_types : list[list[str]]
        -   List containing two sublists consisting of the channel types of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        -   If the types of each channel in a seed/target for a given node are
            identical, this type is given as a string, otherwise the unique
            types are taken and joined into a single string by the " & "
            characters.
        """

        ch_reref_types = {}
        for node_i, combined_names in enumerate(self.results[0].names):
            ch_reref_types[combined_names] = ordered_list_from_dict(
                list_order=self._separated_names[node_i],
                dict_to_order=self.extra_info["ch_reref_types"],
            )
            unique_types = unique(ch_reref_types[combined_names])
            if len(unique_types) > 1:
                unique_types = [combine_vals_list(unique_types)]
            ch_reref_types[combined_names] = unique_types[0]

        node_reref_types = [[], []]
        for group_i in range(2):
            for name in self.extra_info["node_ch_names"][group_i]:
                node_reref_types[group_i].append(ch_reref_types[name])

        return node_reref_types

    def _generate_node_ch_coords(self) -> list[list[list[int, float]]]:
        """Gets the coordinates of channels in the connectivity results.

        RETURNS
        -------
        node_ch_coords : list[list[list[int | float]]]
        -   List containing two sublists consisting of the channel coordinates
            of the seeds and targets, respectively, for each node in the
            connectivity results, with each entry within the sublists being the
            coordinates for each channel in the node group.
        """

        ch_coords = {}
        for node_i, combined_names in enumerate(self.results[0].names):
            ch_coords[combined_names] = [
                self.signal.get_coordinates(name)[0]
                for name in self._separated_names[node_i]
            ]

        node_ch_coords = [[], []]
        for group_i in range(2):
            for name in self.extra_info["node_ch_names"][group_i]:
                node_ch_coords[group_i].append(ch_coords[name])

        return node_ch_coords

    def _generate_node_ch_regions(self) -> list[list[str]]:
        """Gets the regions of channels in the connectivity results.

        RETURNS
        -------
        node_ch_regions : list[list[str]]
        -   List containing two sublists consisting of the channel regions of
            the seeds and targets, respectively, for each node in the
            connectivity results.
        -   If the regions of each channel in a seed/target for a given node are
            identical, this regions is given as a string, otherwise the unique
            regions are taken and joined into a single string by the " & "
            characters.
        """

        ch_regions = {}
        for node_i, combined_names in enumerate(self.results[0].names):
            ch_regions[combined_names] = ordered_list_from_dict(
                list_order=self._separated_names[node_i],
                dict_to_order=self.extra_info["ch_regions"],
            )
            unique_types = unique(ch_regions[combined_names])
            if len(unique_types) > 1:
                unique_types = [combine_vals_list(unique_types)]
            ch_regions[combined_names] = unique_types[0]

        node_ch_regions = [[], []]
        for group_i in range(2):
            for name in self.extra_info["node_ch_names"][group_i]:
                node_ch_regions[group_i].append(ch_regions[name])

        return node_ch_regions

    def _generate_node_ch_hemispheres(
        self,
    ) -> tuple[list[list[str]], list[list[bool]]]:
        """Gets the hemispheres of channels in the connectivity results.

        RETURNS
        -------
        node_ch_hemispheres : list[list[str]]
        -   List containing two sublists consisting of the hemispheres of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        -   If the hemispheres of each channel in a seed/target for a given node
            are identical, this hemispheres is given as a string, otherwise the
            unique hemispheres are taken and joined into a single string by the
            " & " characters.

        node_single_hemispheres : list[list[bool]]
        -   list containing two sublists of bools stating whether the channels
            in the seeds/targets of each node were derived from the same
            hemisphere.
        """

        ch_hemispheres = {}
        single_hemispheres = {name: True for name in self.results[0].names}
        for node_i, combined_names in enumerate(self.results[0].names):
            ch_hemispheres[combined_names] = ordered_list_from_dict(
                list_order=self._separated_names[node_i],
                dict_to_order=self.extra_info["ch_hemispheres"],
            )
            unique_types = unique(ch_hemispheres[combined_names])
            if len(unique_types) > 1:
                unique_types = [combine_vals_list(unique_types)]
                single_hemispheres[combined_names] = False
            ch_hemispheres[combined_names] = unique_types[0]

        node_ch_hemispheres = [[], []]
        node_single_hemispheres = [[], []]
        for group_i in range(2):
            for name in self.extra_info["node_ch_names"][group_i]:
                node_ch_hemispheres[group_i].append(ch_hemispheres[name])
                node_single_hemispheres[group_i].append(
                    single_hemispheres[name]
                )

        return node_ch_hemispheres, np.asarray(node_single_hemispheres)

    def _generate_node_lateralisation(
        self, node_single_hemispheres: list[list[bool]]
    ) -> list[str]:
        """Gets the lateralisation of the channels in the connectivity node.
        -   Can either be "contralateral" if the seed and target are from
            different hemispheres, or "ipsilateral" if the seed and target are
            from the same hemisphere.

        PARAMETERS
        ----------
        node_single_hemispheres : list[list[bool]]
        -   list containing two sublists of bools stating whether the channels
            in the seeds/targets of each node were derived from the same
            hemisphere.

        RETURNS
        -------
        node_lateralisation : list[str]
        -   Lateralisation ("ipsilateral & contralateral", "contralateral", or
            "ipsilateral") of each connectivity node.
        """

        node_lateralisation = []
        node_ch_hemispheres = self.extra_info["node_ch_hemispheres"]
        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                if (
                    not node_single_hemispheres[0][node_i]
                    or not node_single_hemispheres[1][node_i]
                ):
                    lateralisation = "ipsilateral & contralateral"
                else:
                    lateralisation = "contralateral"
            else:
                lateralisation = "ipsilateral"
            node_lateralisation.append(lateralisation)

        return node_lateralisation

    def _generate_node_ch_epoch_orders(self) -> list[list[str]]:
        """Gets the epoch orders of channels in the connectivity results.

        RETURNS
        -------
        node_epoch_orders : list[str]
        -   List containing the epoch orders of the seeds and targets for each
            node in the connectivity results. If either the seed or target has
            a "shuffled" epoch order, the epoch order of the node is "shuffled",
            otherwise it is "original".
        """

        ch_epoch_orders = {}
        for node_i, combined_names in enumerate(self.results[0].names):
            ch_epoch_orders[combined_names] = ordered_list_from_dict(
                list_order=self._separated_names[node_i],
                dict_to_order=self.extra_info["ch_epoch_orders"],
            )
            unique_types = unique(ch_epoch_orders[combined_names])
            if len(unique_types) > 1:
                unique_types = [combine_vals_list(unique_types)]
            ch_epoch_orders[combined_names] = unique_types[0]

        node_epoch_orders = []
        for node_i in range(len(self.extra_info["node_ch_names"][0])):
            seed_names = self.extra_info["node_ch_names"][0][node_i]
            target_names = self.extra_info["node_ch_names"][1][node_i]
            if (
                ch_epoch_orders[seed_names] == "original"
                and ch_epoch_orders[target_names] == "original"
            ):
                order = "original"
            else:
                order = "shuffled"
            node_epoch_orders.append(order)

        return node_epoch_orders

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   Whether or not the user is asked to confirm to overwrite a
            pre-existing file if one exists.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_object(
            to_save=self,
            fpath=fpath,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def save_results(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the results and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved.

        ftype : str | None; default None
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_dict(
            to_save=self.results_as_dict(),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def results_as_dict(self) -> dict:
        """Returns the results and additional information as a dictionary.

        RETURNS
        -------
        dict
        -   The results and additional information stored as a dictionary.
        """

        dimensions = self._get_optimal_dims()
        results = self.get_results(dimensions=dimensions)

        return {
            f"connectivity-{self._method}": results.tolist(),
            f"connectivity-{self._method}_dimensions": dimensions,
            "freqs": self.results[0].freqs,
            "seed_names": self.extra_info["node_ch_names"][0],
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self.extra_info["node_ch_names"][1],
            "target_types": self.extra_info["node_ch_types"][1],
            "target_coords": self.extra_info["node_ch_coords"][1],
            "target_regions": self.extra_info["node_ch_regions"][1],
            "target_hemispheres": self.extra_info["node_ch_hemispheres"][1],
            "target_reref_types": self.extra_info["node_ch_reref_types"][1],
            "node_lateralisation": self.extra_info["node_lateralisation"],
            "node_epoch_orders": self.extra_info["node_ch_epoch_orders"],
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }

    def get_results(self, dimensions: Union[list[str], None] = None) -> NDArray:
        """Extracts and returns results.

        PARAMETERS
        ----------
        dimensions : list[str] | None;  default None
        -   The dimensions of the results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        results : numpy array
        -   The results.
        """

        if dimensions is None:
            dimensions = self.results_dims

        if self._windows_averaged:
            results = self.results[0].get_data()
        else:
            results = []
            for mne_obj in self.results:
                results.append(mne_obj.get_data())
            results = np.asarray(results)

        results = rearrange_axes(
            obj=results, old_order=self.results_dims, new_order=dimensions
        )

        return deepcopy(results)
