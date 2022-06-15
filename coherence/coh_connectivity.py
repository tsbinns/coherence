"""Classes for calculating connectivity between signals.

CLASSES
-------
ConnectivityCoherence : subclass of the abstract base class 'ProcMethod'
-   Calculates the coherence (standard or imaginary) between signals.

ConectivityMultivariate : subclass of the abstract base class 'ProcMethod'
-   Calculates the multivariate connectivity (multivariate interaction measure,
    MIM, or maximised imaginary coherence, MIC) between signals.
"""

from typing import Optional, Union
from mne import Epochs
from mne.time_frequency import (
    CrossSpectralDensity,
    csd_fourier,
    csd_multitaper,
    csd_morlet,
)
from mne_connectivity import (
    seed_target_indices,
    spectral_connectivity_epochs,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
)
from numpy.typing import NDArray
import numpy as np
from coh_connectivity_computations import (
    granger_causality,
    multivariate_connectivity,
)
from coh_exceptions import (
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_processing_methods import ProcConnectivity
from coh_progress_bar import ProgressBar
import coh_signal
from coh_saving import save_dict


class ConnectivityCoherence(ProcConnectivity):
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
        super()._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._con_method = None
        self._pow_method = None
        self._seeds = None
        self._targets = None
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
        self._progress_bar = None

        # Initialises aspects of the object that indicate which methods have
        # been called (starting as 'False'), which can later be updated.
        self._timepoints_averaged = False

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

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""
        if self._verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data) * len(self._indices),
                title="Computing connectivity",
            )

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
                    method=self._con_method,
                    indices=self._indices,
                    sfreq=data.info["sfreq"],
                    mode=self._pow_method,
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
            if self._con_method == "imcoh":
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
            if self._progress_bar is not None:
                self._progress_bar.update_progress()
        self.results = connectivity

        if self._progress_bar is not None:
            self._progress_bar.close()

        self._sort_dimensions()
        super()._generate_extra_info()

    def process(
        self,
        con_method: str,
        pow_method: str,
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
        con_method : str
        -   The method for calculating connectivity.
        -   Supported inputs are: 'coh' - standard coherence; 'cohy' -
            coherency; 'imcoh' - imaginary part of coherence; 'plv' -
            phase-locking value; 'ciplv' - corrected imaginary phase-locking
            value; 'ppc' - pairwise phase consistency; 'pli' - phase lag index;
            'pli2_unbiased' - unbiased estimator of squared phase lag index;
            'wpli' - weighted phase lag index; 'wpli2_debiased' - debiased
            estimator of squared weighted phase lag index.

        pow_method : str
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

        self._con_method = con_method
        self._pow_method = pow_method
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

        super()._sort_seeds_targets()

        self._get_results()

        self._processed = True
        self.processing_steps["connectivity_coherence"] = {
            "con_method": con_method,
            "pow_method": pow_method,
            "seeds": self._seeds_str,
            "targets": self._targets_str,
            "fmin": fmin,
            "fmax": fmax,
            "fskip": fskip,
            "faverage": faverage,
            "tmin": tmin,
            "tmax": tmax,
            "mt_bandwidth": mt_bandwidth,
            "mt_adaptive": mt_adaptive,
            "mt_low_bias": mt_low_bias,
            "cwt_freqs": self._cwt_freqs,
            "cwt_n_cycles": cwt_n_cycles,
            "average_windows": average_windows,
            "average_timepoints": average_timepoints,
        }

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
            f"connectivity-{self._con_method}": results.tolist(),
            f"connectivity-{self._con_method}_dimensions": dimensions,
            "freqs": self.results[0].freqs,
            "seed_names": self._seeds_str,
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self._targets_str,
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


class ConnectivityMultivariate(ProcConnectivity):
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
        super()._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._con_method = None
        self._cohy_method = None
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
        self._return_topographies = None
        self._block_size = None
        self._n_jobs = None
        self._cohy_matrix_indices = None
        self.seed_topographies = None
        self.target_topographies = None
        self._progress_bar = None

    def process(
        self,
        con_method: str,
        cohy_method: str,
        seeds: Union[str, list[str], None] = None,
        targets: Union[str, list[str], None] = None,
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
        return_topographies: bool = False,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        spectral_connectivity_epochs function of the mne-connectivity package to
        generate coherency values for the computation of multivariate
        connectivity metrics.

        PARAMETERS
        ----------
        con_method : str
        -   The multivariate connectivity metric to compute.
        -   Supported inputs are: 'mim' - multivariate interaction measure; and
            'mic' - maximised imaginary coherence.

        cohy_method : str
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

        cwt_freqs: numpy array[int | float] | None
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

        return_topographies: bool; default False
        -   Whether or not to return spatial topographies of connectivity for
            the signals.
        -   Only available when calculating maximised imaginary coherence.

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

        self._con_method = con_method
        self._cohy_method = cohy_method
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
        if return_topographies and self._con_method == "mic":
            self._return_topographies = return_topographies
        else:
            self._return_topographies = False

        self._sort_seeds_targets()

        self._get_results()

        self._processed = True
        self.processing_steps["connectivity_multivariate"] = {
            "con_method": con_method,
            "cohy_method": cohy_method,
            "seeds": self._seeds_str,
            "targets": self._targets_str,
            "fmin": fmin,
            "fmax": fmax,
            "fskip": fskip,
            "faverage": faverage,
            "tmin": tmin,
            "tmax": tmax,
            "mt_bandwidth": mt_bandwidth,
            "mt_adaptive": mt_adaptive,
            "mt_low_bias": mt_low_bias,
            "cwt_freqs": self._cwt_freqs,
            "cwt_n_cycles": cwt_n_cycles,
            "average_windows": average_windows,
            "return_topographies": self._return_topographies,
        }

    def _sort_seeds_targets(self) -> None:
        """Sorts seeds and targets for the connectivity analysis."""
        super()._sort_seeds_targets()
        self._sort_cohy_indices()

    def _sort_cohy_indices(self) -> None:
        """Gets the indices of the seeds and targets for each connectivity node
        that will be taken from the coherency matrix."""
        self._cohy_matrix_indices = []
        for seeds, targets in zip(self._seeds_list, self._targets_list):
            ch_names = [*seeds, *targets]
            ch_idcs = [
                self.signal.data[0].ch_names.index(name) for name in ch_names
            ]
            self._cohy_matrix_indices.append(
                seed_target_indices(ch_idcs, ch_idcs)
            )

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""
        if self._verbose:
            self._progress_bar = ProgressBar(
                n_steps=(len(self._indices[0]) + 1) * len(self.signal.data),
                title="Computing connectivity",
            )

        connectivity = []
        topographies = [[], []]
        for win_i, win_data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"Computing connectivity for window {win_i+1} of "
                    f"{len(self.signal.data)}.\n"
                )
            coherency = self._get_cohy(win_data)
            con_results, topo_results = self._get_multivariate_results(
                coherency
            )
            connectivity.append(con_results)
            topographies[0].append(topo_results[0])
            topographies[1].append(topo_results[1])
        self.results = connectivity
        if self._return_topographies:
            self.seed_topographies = np.asarray(topographies[0], dtype=object)
            self.target_topographies = np.asarray(topographies[1], dtype=object)

        if self._progress_bar is not None:
            self._progress_bar.close()

        self._sort_dimensions()
        super()._generate_extra_info()

    def _get_cohy(self, data: Epochs) -> SpectralConnectivity:
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
        coherency : MNE SpectralConnectivity
        -   The coherency between all channels in the data.
        """
        coherency = []
        if self._verbose:
            print("Computing coherency for the data.")

        coherency = spectral_connectivity_epochs(
            data=data,
            method="cohy",
            indices=seed_target_indices(
                np.arange(len(data.ch_names)), np.arange(len(data.ch_names))
            ),
            sfreq=data.info["sfreq"],
            mode=self._cohy_method,
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
        if isinstance(coherency, SpectroTemporalConnectivity):
            coherency = SpectralConnectivity(
                data=np.mean(coherency.get_data(), axis=-1),
                freqs=coherency.freqs,
                n_nodes=coherency.n_nodes,
                names=coherency.names,
                indices=coherency.indices,
                method=coherency.method,
                n_epochs_used=coherency.n_epochs_used,
            )
        if self._progress_bar is not None:
            self._progress_bar.update_progress()

        return coherency

    def _get_multivariate_results(self, data: SpectralConnectivity) -> None:
        """For a single window, computes the multivariate connectivity results
        using the coherency data.

        PARAMETERS
        ----------
        data : MNE SpectralConnectivity
        -   The coherency data for a single window.

        RETURNS
        -------
        results : MNE SpectralConnectivity
        -   The multivariate connectivity results of a single window for all
            seed-target pairs.

        topographies : list[list[numpy array]]
        -   The spatial topographies of seeds and targets, respectively, for all
            seed-target pairs in a single window.
        """
        connectivity = []
        topographies = [[], []]
        for con_i, indices in enumerate(self._cohy_matrix_indices):
            if self._verbose:
                print(
                    f"Computing '{self._con_method}' for seed-target group "
                    f"{con_i+1} of {len(self._indices[0])}.\n"
                )
            cohy_matrix = self._get_cohy_matrix(data, indices)
            results = multivariate_connectivity(
                data=cohy_matrix,
                method=self._con_method,
                n_group_a=len(self._seeds_list[con_i]),
                n_group_b=len(self._targets_list[con_i]),
                return_topographies=self._return_topographies,
            )
            if self._return_topographies and self._con_method == "mic":
                connectivity.append(results[0])
                topographies[0].append(results[1][0])
                topographies[1].append(results[1][1])
            else:
                connectivity.append(results)
            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        results = self._multivariate_to_mne(
            data=np.asarray(connectivity),
            freqs=data.freqs,
            n_epochs_used=data.n_epochs_used,
        )

        return results, topographies

    def _get_cohy_matrix(
        self, data: SpectralConnectivity, indices: list[list[int]]
    ) -> NDArray:
        """Converts the coherency data from a two-dimensional matrix into into a
        three-dimensional matrix with dimensions [nodes x nodes x frequencies],
        where the nodes are picked based on the specified indices.

        PARAMETERS
        ----------
        data : MNE SpectralConnectivity
        -   MNE connectivity object containing the coherency values between all
            channels.

        indices : list[list[int]]
        -   Indices of the seeds and targets that should be picked from 'data'
            for conversion into the three-dimensional matrix.

        RETURNS
        -------
        data_matrix : numpy Array
        -   A three-dimensional array containing the coherency values for all
            possible connections between the seed-target pairs in the first two
            dimensions, and frequencies in the third dimension.
        """
        node_idcs = []
        node_idx = 0
        for seed_idx_data, target_idx_data in zip(
            data.indices[0], data.indices[1]
        ):
            for seed_idx_indices, target_idx_indices in zip(
                indices[0], indices[1]
            ):
                if (
                    seed_idx_data == seed_idx_indices
                    and target_idx_data == target_idx_indices
                ):
                    node_idcs.append(node_idx)
            node_idx += 1

        data_vals = data.get_data()
        n_nodes = int(np.sqrt(len(indices[0])))
        n_freqs = len(data.freqs)
        data_matrix = np.empty((n_nodes, n_nodes, n_freqs), dtype="complex128")
        for freq_i in range(n_freqs):
            data_matrix[:, :, freq_i] = np.reshape(
                data_vals[node_idcs, freq_i], (n_nodes, n_nodes)
            )

        return data_matrix

    def _multivariate_to_mne(
        self, data: NDArray, freqs: NDArray, n_epochs_used: int
    ) -> SpectralConnectivity:
        """Converts results of the multivariate connectivity analysis stored as
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
        return SpectralConnectivity(
            data=data,
            freqs=freqs,
            n_nodes=len(self._seeds_list),
            names=self._comb_names_str,
            indices=self._indices,
            method=self._con_method,
            n_epochs_used=n_epochs_used,
        )

    def _sort_dimensions(self) -> None:
        """Establishes dimensions of the connectivity results and averages
        across windows, if requested."""

        self._results_dims = ["windows", "channels", "frequencies"]

        if self._average_windows:
            self._average_windows_results()

        if self._return_topographies:
            groups = ["seed_topographies", "target_topographies"]
            for group in groups:
                win_topos = []
                for win_data in getattr(self, group)[0]:
                    node_topos = []
                    for node_data in win_data:
                        node_topos.append(node_data.tolist())
                    win_topos.append(node_topos)
                setattr(self, group, [win_topos])

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
        if self._return_topographies:
            self.seed_topographies = [np.mean(self.seed_topographies, axis=0)]
            self.target_topographies = [
                np.mean(self.target_topographies, axis=0)
            ]
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
        con_results, topo_results = self.get_results(dimensions=dimensions)

        results_dict = {
            f"connectivity-{self._con_method}": con_results.tolist(),
            f"connectivity-{self._con_method}_dimensions": dimensions,
            "freqs": self.results[0].freqs,
            "seed_names": self._seeds_str,
            "seed_topographies": topo_results[0],
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self._targets_str,
            "target_topographies": topo_results[1],
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

        if not self._return_topographies:
            del results_dict["seed_topographies"]
            del results_dict["target_topographies"]

        return results_dict

    def get_results(
        self, dimensions: Union[list[str], None] = None
    ) -> tuple[NDArray, list[Union[list, None]]]:
        """Gets the connectivity and topography results of the analysis.

        PARAMETERS
        ----------
        dimensions : list[str] | None; default None
        -   The dimensions of the connectivity results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        numpy array
        -   The connectivity results.

        list[list | None]
        -   The topographies of the connectivity results.
        -   If topographies have been computed, a list with two sublists
            containing the topographies for the seed and target channels,
            respectively, for each connectivity node are returned, in which case
            the topographies have dimensions [windows x nodes x channels x
            frequencies] if windows have not been averaged over, or [nodes x
            channels x frequencies] if they have.
        -   If topographies have not been calculated, a list with two 'None'
            entries is returned.
        """

        return (
            super().get_results(dimensions=dimensions),
            self._get_topography_results(),
        )

    def _get_topography_results(self) -> list[Union[list, None]]:
        """Gets the topography results.

        RETURNS
        -------
        topographies : list[list | None]
        -   The topographies of the connectivity results.
        -   If topographies have been computed, a list with two sublists
            containing the topographies for the seed and target channels,
            respectively, for each connectivity node are returned, in which case
            the topographies have dimensions [windows x nodes x channels x
            frequencies] if windows have not been averaged over, or [nodes x
            channels x frequencies] if they have.
        -   If topographies have not been calculated, a list with two 'None'
            entries is returned.
        """

        if self._return_topographies:
            topographies = []
            if self._average_windows:
                topographies.append(self.seed_topographies[0])
                topographies.append(self.target_topographies[0])
            else:
                topographies.append(self.seed_topographies)
                topographies.append(self.target_topographies)
        else:
            topographies = [None, None]

        return topographies


class ConnectivityGranger(ProcConnectivity):
    """Calculates Granger causality measures of connectivity between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs granger causality analysis.

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
        super()._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._gc_method = None
        self._cs_method = None
        self._seeds = None
        self._targets = None
        self._n_lags = None
        self._cwt_freqs = None
        self._fmt_fmin = None
        self._fmt_fmax = None
        self._tmin = None
        self._tmax = None
        self._picks = None
        self._cwt_n_cycles = None
        self._cwt_use_fft = None
        self._fmt_n_fft = None
        self._mt_bandwidth = None
        self._mt_adaptive = None
        self._mt_low_bias = None
        self._cwt_decim = None
        self._average_windows = None
        self._n_jobs = None
        self._progress_bar = None
        self._csd_matrix_indices = None
        self._freqs = None

    def process(
        self,
        gc_method: str,
        cs_method: str,
        seeds: Union[str, list[str], None] = None,
        targets: Union[str, list[str], None] = None,
        n_lags: int = 20,
        tmin: Union[int, float, None] = None,
        tmax: Union[int, float, None] = None,
        average_windows: bool = True,
        n_jobs: int = 1,
        cwt_freqs: Union[list[Union[int, float]], None] = None,
        cwt_n_cycles: Union[int, float, list[Union[int, float]]] = 7,
        cwt_use_fft: bool = True,
        cwt_decim: Union[int, slice] = 1,
        mt_bandwidth: Union[int, float, None] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        fmt_fmin: Union[int, float] = 0,
        fmt_fmax: Union[int, float] = float("inf"),
        fmt_n_fft: Union[int, None] = None,
    ):
        """Performs the Granger casuality analysis on the data.

        PARAMETERS
        ----------
        gc_method : str
        -   The Granger causality metric to compute.
        -   Supported inputs are: "gc" - standard Granger causality; and "trgc"
            - time-reversed Granger causality.

        cs_method : str
        -   The method for computing the cross-spectra of the data.
        -   Supported inputs are: "multitaper"; "fourier"; and "cwt_morlet".

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

        n_lags : int; default 20
        -   The number of lags to use when computing autocovariance. Currently,
            only positive-valued integers are supported.

        tmin : float | None; default None
        -   Time to start the connectivity estimation.
        -   If None, the data is used from the beginning.

        tmax : float | None; default None
        -   Time to end the connectivity estimation.
        -   If None, the data is used until the end.

        average_windows : bool; default False
        -   Whether or not to average connectivity results across windows.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.

        cwt_freqs : list[int | float] | None; default None
        -   The frequencies of interest, in Hz.
        -   Only used if 'cs_method' is "cwt_morlet", in which case 'freqs' cannot
            be 'None'.

        cwt_n_cycles: int | float | array[int | float]; default 7
        -   The number of cycles to use when calculating connectivity.
        -   If an single integer or float, this number of cycles is for each
            frequency.
        -   If an array, the entries correspond to the number of cycles to use
            for each frequency being analysed.
        -   Only used if 'cs_method' is "cwt_morlet".

        cwt_use_fft : bool; default True
        -   Whether or not FFT-based convolution is used to compute the wavelet
            transform.
        -   Only used if 'cs_method' is "cwt_morlet".

        cwt_decim : int | slice; default 1
        -   Decimation factor to use during time-frequency decomposition to
            reduce memory usage. If 1, no decimation is performed.

        mt_bandwidth : float | None; default None
        -   The bandwidth, in Hz, of the multitaper windowing function.
        -   Only used if 'cs_method' is "multitaper".

        mt_adaptive : bool; default False
        -   Whether or not to use adaptive weights to comine the tapered spectra
            into power spectra.
        -   Only used if 'cs_method' is "multitaper".

        mt_low_bias : bool: default True
        -   Whether or not to only use tapers with > 90% spectral concentration
            within bandwidth.
        -   Only used if 'cs_method' is "multitaper".

        fmt_fmin : int | float; default 0
        -   The lower frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            lower frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their lower frequencies.
        -   Only used if 'cs_method' is "fourier" or "multitaper".

        fmt_fmax : int | float; default infinity
        -   The higher frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            higher frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their higher frequencies.
        -   If infinity, no higher frequency is used.
        -   Only used if 'cs_method' is "fourier" or "multitaper".

        fmt_n_fft : int | None; default None
        -   Length of the FFT.
        -   If 'None', the number of samples between 'tmin' and 'tmax' is used.
        -   Only used if 'cs_method' is "fourier" or "multitaper".
        """
        self._gc_method = gc_method
        self._cs_method = cs_method
        self._seeds = seeds
        self._targets = targets
        self._n_lags = n_lags
        self._cwt_freqs = cwt_freqs
        self._fmt_fmin = fmt_fmin
        self._fmt_fmax = fmt_fmax
        self._tmin = tmin
        self._tmax = tmax
        self._cwt_n_cycles = cwt_n_cycles
        self._cwt_use_fft = cwt_use_fft
        self._fmt_n_fft = fmt_n_fft
        self._mt_bandwidth = mt_bandwidth
        self._mt_adaptive = mt_adaptive
        self._mt_low_bias = mt_low_bias
        self._cwt_decim = cwt_decim
        self._average_windows = average_windows
        self._n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

    def _sort_processing_inputs(self) -> None:
        """Checks that inputs for processing the data are appropriate."""
        supported_cs_methods = ["fourier", "multitaper", "cwt_morlet"]
        if self._cs_method not in supported_cs_methods:
            raise NotImplementedError(
                "Error when performing Granger causality analysis:\nThe method "
                f"for computing the cross-spectral density '{self._cs_method}' "
                "is not recognised. Supported inputs are "
                f"{supported_cs_methods}."
            )
        super()._sort_seeds_targets()
        self._sort_csd_indices()

    def _sort_csd_indices(self) -> None:
        """Gets the indices of the seeds and targets for each connectivity node
        that will be taken from the cross-spectral density matrix."""
        self._csd_matrix_indices = [[], []]
        for seeds, targets in zip(self._seeds_list, self._targets_list):
            self._csd_matrix_indices[0].append(
                [self.signal.data[0].ch_names.index(name) for name in seeds]
            )
            self._csd_matrix_indices[1].append(
                [self.signal.data[0].ch_names.index(name) for name in targets]
            )

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""
        if self._verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data) * len(self._indices) * 3,
                title="Computing connectivity",
            )

        cross_spectra = self._compute_csd()
        self._compute_gc(cross_spectra=cross_spectra)

        self._sort_dimensions()
        super()._generate_extra_info()

    def _compute_csd(self) -> list[CrossSpectralDensity]:
        """Computes the cross-spectral density of the data.

        RETURNS
        -------
        cross_spectra : list[MNE CrossSpectralDensity]
        -   The cross-spectra for each window of the data.
        """
        cross_spectra = []
        for i, data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    "Computing the cross-spectral density for window "
                    f"{i+1} of {len(self.signal.data)}.\n"
                )
            if self._cs_method == "fourier":
                cross_spectra.append(
                    csd_fourier(
                        epochs=data,
                        fmin=self._fmt_fmin,
                        fmax=self._fmt_fmax,
                        tmin=self._tmin,
                        tmax=self._tmax,
                        n_fft=self._fmt_n_fft,
                        projs=None,
                        n_jobs=self._n_jobs,
                        verbose=self._verbose,
                    )
                )
            elif self._cs_method == "multitaper":
                cross_spectra.append(
                    csd_multitaper(
                        epochs=data,
                        fmin=self._fmt_fmin,
                        fmax=self._fmt_fmax,
                        tmin=self._tmin,
                        tmax=self._tmax,
                        n_fft=self._fmt_n_fft,
                        bandwidth=self._mt_bandwidth,
                        adaptive=self._mt_adaptive,
                        low_bias=self._mt_low_bias,
                        projs=None,
                        n_jobs=self._n_jobs,
                        verbose=self._verbose,
                    )
                )
            elif self._cs_method == "cwt_morlet":
                cross_spectra.append(
                    csd_morlet(
                        epochs=data,
                        frequencies=self._cwt_freqs,
                        tmin=self._tmin,
                        tmax=self._tmax,
                        n_cycles=self._cwt_n_cycles,
                        use_fft=self._cwt_use_fft,
                        decim=self._cwt_decim,
                        projs=None,
                        n_jobs=self._n_jobs,
                        verbose=self._verbose,
                    )
                )
            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        self._freqs = cross_spectra[0].frequencies

        return cross_spectra

    def _compute_gc(self, cross_spectra: list[CrossSpectralDensity]) -> None:
        """Computes Granger casuality between signals from the cross-spectral
        density for each window.

        PARAMETERS
        ----------
        cross_spectra : list[MNE CrossSpectralDensity]
        -   The cross-spectra between signals for each window.
        """
        results = []
        for i, csd in enumerate(cross_spectra):
            if self._verbose:
                print(
                    f"Computing Granger causality for window {i+1} of "
                    f"{len(self.signal.data)}.\n"
                )
            csd_matrix = np.transpose(
                np.asarray(
                    [csd.get_data(frequency=freq) for freq in self._freqs]
                ),
                (1, 2, 0),
            )
            results.append(
                granger_causality(
                    csd=csd_matrix,
                    freqs=self._freqs,
                    method=self._gc_method,
                    seeds=self._csd_matrix_indices[0],
                    targets=self._csd_matrix_indices[1],
                    n_lags=self._n_lags,
                )
            )
            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        self.results = results

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
        self.results = [
            SpectralConnectivity(
                data=np.asarray([data for data in self.results]).mean(axis=0),
                freqs=self._freqs,
                n_nodes=len(self._comb_names_str),
                names=self._comb_names_str,
                indices=self._indices,
                method=self._gc_method,
                n_epochs_used=self.signal.data[0]._data.shape[0],
            )
        ]

        self._windows_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

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
        results = super().get_results(dimensions=dimensions)

        return {
            f"connectivity-{self._gc_method}": results.tolist(),
            f"connectivity-{self._gc_method}_dimensions": dimensions,
            "freqs": self._freqs,
            "seed_names": self._seeds_str,
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self._targets_str,
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
