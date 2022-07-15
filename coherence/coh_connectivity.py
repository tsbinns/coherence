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
from typing import Union
from mne import Epochs
from mne.time_frequency import (
    CrossSpectralDensity,
    csd_array_fourier,
    csd_array_multitaper,
    csd_array_morlet,
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
from coh_handle_entries import rearrange_axes
from coh_connectivity_processing_methods import (
    ProcMultivariateConnectivity,
    ProcSingularConnectivity,
)
from coh_progress_bar import ProgressBar
import coh_signal
from coh_saving import save_dict


class ConnectivityCoherence(ProcSingularConnectivity):
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
        if self._pow_method in ["multitaper", "fourier"]:
            self._results_dims.extend(["channels", "frequencies"])
        elif self._pow_method in ["cwt_morlet"]:
            self._results_dims.extend(["channels", "frequencies", "timepoints"])
        else:
            raise UnavailableProcessingError(
                "Error when sorting the results of the connectivity analysis:\n"
                f"The analysis mode '{self._pow_method}' does not have an associated "
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
        seeds: Union[Union[str, list[str]], None] = None,
        targets: Union[Union[str, list[str]], None] = None,
        fmin: Union[Union[float, tuple], None] = None,
        fmax: Union[float, tuple] = float("inf"),
        fskip: int = 0,
        faverage: bool = False,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        mt_bandwidth: Union[float, None] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Union[Union[list, NDArray], None] = None,
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
            "seeds": self._seeds,
            "targets": self._targets,
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
        ftype: Union[str, None] = None,
        ask_before_overwrite: Union[bool, None] = None,
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
            "seed_names": self._seeds,
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_subregions": self.extra_info["node_ch_subregions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self._targets,
            "target_types": self.extra_info["node_ch_types"][1],
            "target_coords": self.extra_info["node_ch_coords"][1],
            "target_regions": self.extra_info["node_ch_regions"][1],
            "target_subregions": self.extra_info["node_ch_subregions"][1],
            "target_hemispheres": self.extra_info["node_ch_hemispheres"][1],
            "target_reref_types": self.extra_info["node_ch_reref_types"][1],
            "node_lateralisation": self.extra_info["node_lateralisation"],
            "node_epoch_orders": self.extra_info["node_ch_epoch_orders"],
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }


class ConnectivityMIMMIC(ProcMultivariateConnectivity):
    """Calculates the multivariate connectivity measures multivariate
    interaction measure (MIM) or maximised imaginary coherence (MIC) between
    signals.

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
        self._topography_results_dims = None
        self._progress_bar = None

    def process(
        self,
        con_method: str,
        cohy_method: str,
        seeds: Union[str, list[str], None] = None,
        targets: Union[str, list[str], None] = None,
        fmin: Union[Union[float, tuple], None] = None,
        fmax: Union[float, tuple] = float("inf"),
        fskip: int = 0,
        faverage: bool = False,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        mt_bandwidth: Union[float, None] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Union[Union[list, NDArray], None] = None,
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
                n_seeds=len(self._seeds_list[con_i]),
                n_targets=len(self._targets_list[con_i]),
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

        self._results_dims = ["windows", "nodes", "frequencies"]
        self._topography_results_dims = [
            "windows",
            "nodes",
            "channels",
            "frequencies",
        ]

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

    @property
    def topography_results_dims(self) -> list[str]:
        """Returns the dimensions of the spatial topography results.

        RETURNS
        -------
        dims : list[str]
        -   Dimensions of the results.
        """
        if self._windows_averaged:
            dims = self._topography_results_dims[1:]
        else:
            dims = self._topography_results_dims

        return deepcopy(dims)

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        ask_before_overwrite: Union[bool, None] = None,
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

        to_save = self.results_as_dict()
        if self._return_topographies:
            save_dict(
                to_save=to_save[0],
                fpath=fpath,
                ftype=ftype,
                ask_before_overwrite=ask_before_overwrite,
                verbose=self._verbose,
            )
            save_dict(
                to_save=to_save[1],
                fpath=f"{fpath}_topography",
                ftype=ftype,
                ask_before_overwrite=ask_before_overwrite,
                verbose=self._verbose,
            )
        else:
            save_dict(
                to_save=to_save,
                fpath=fpath,
                ftype=ftype,
                ask_before_overwrite=ask_before_overwrite,
                verbose=self._verbose,
            )

    def results_as_dict(self) -> Union[dict, tuple[dict, dict]]:
        """Returns the connectivity and, if applicable, topography results, as
        well as additional information as a dictionary.

        RETURNS
        -------
        results : dict | tuple(dict, dict)
        -   The results and additional information stored as a dictionary.
        -   If spatial topographies have not been computed, a single dict
            containing the connectivity results is returned, otherwise a tuple
            of two dicts is returned containing the connectivity and spatial
            topography results, respectively.
        """

        connectivity_results = self.connectivity_results_as_dict()
        if self._return_topographies:
            topography_results = self.topography_results_as_dict()
            results = (connectivity_results, topography_results)
        else:
            results = connectivity_results

        return results

    def connectivity_results_as_dict(self) -> dict:
        """Returns the connectivity results and additional information as a
        dictionary.

        RETURNS
        -------
        dict
        -   The results and additional information stored as a dictionary.
        """

        dimensions = self._get_optimal_dims()
        results = self.get_connectivity_results(dimensions=dimensions)

        return {
            f"connectivity-{self._con_method}": results.tolist(),
            f"connectivity-{self._con_method}_dimensions": dimensions,
            "freqs": self.results[0].freqs,
            "seed_names": self._seeds_str,
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_subregions": self.extra_info["node_ch_subregions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self._targets_str,
            "target_types": self.extra_info["node_ch_types"][1],
            "target_coords": self.extra_info["node_ch_coords"][1],
            "target_regions": self.extra_info["node_ch_regions"][1],
            "target_subregions": self.extra_info["node_ch_subregions"][1],
            "target_hemispheres": self.extra_info["node_ch_hemispheres"][1],
            "target_reref_types": self.extra_info["node_ch_reref_types"][1],
            "node_lateralisation": self.extra_info["node_lateralisation"],
            "node_epoch_orders": self.extra_info["node_ch_epoch_orders"],
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }

    def topography_results_as_dict(self) -> dict:
        """Returns the topography results and additional information as a
        dictionary.

        The form of the results differs from the connectivity results, in that
        each entry represents not a connectivity node, but a channel within a
        connectivity node.

        RETURNS
        results : dict
        -   The results and additional information stored as a dictionary.

        RAISES
        ------
        AttributeError
        -   Raised if spatial topography results have not been computed.
        """
        if not self._return_topographies:
            raise AttributeError(
                "Spatial topography results have not been computed, and thus "
                "cannot be converted into a dictionary."
            )
        if not self._windows_averaged:
            raise NotImplementedError(
                "Returning spatial topographies when windows have not been "
                "averaged over has not been implemented."
            )

        results, connectivity_results = self._prepare_topography_results_dict()

        topos_name = f"connectivity-{self._con_method}_topography"
        for node_i in range(len(self._seeds_list)):
            for group in ["seed", "target"]:
                if group == "seed":
                    ch_names = self._seeds_list[node_i]
                    topographies = self.seed_topographies[0][node_i]
                else:
                    ch_names = self._targets_list[node_i]
                    topographies = self.target_topographies[0][node_i]
                for ch_i, ch_name in enumerate(ch_names):
                    results[topos_name].append(topographies[ch_i].tolist())
                    results["ch_names"].append(ch_name)
                    results["ch_types"].append(
                        self.signal.data[0].get_channel_types(picks=ch_name)[0]
                    )
                    results["ch_coords"].append(
                        self.signal.get_coordinates(ch_name)[0]
                    )
                    results["ch_regions"].append(
                        self.extra_info["ch_regions"][ch_name]
                    )
                    results["ch_subregions"].append(
                        self.extra_info["ch_subregions"][ch_name]
                    )
                    results["ch_hemispheres"].append(
                        self.extra_info["ch_hemispheres"][ch_name]
                    )
                    results["ch_reref_types"].append(
                        self.extra_info["ch_reref_types"][ch_name]
                    )
                    results["ch_node_types"].append(group)
                    for key, value in connectivity_results.items():
                        results[key].append(value[node_i])

        return results

    def _prepare_topography_results_dict(self) -> tuple[dict, dict]:
        """Prepares dictionaries that will be used for storing the spatial
        topography results as a dictionary.

        RETURNS
        -------
        topography_results : dict
        -   Dictionary partly filled with data independent of connectivity nodes
            (e.g. frequencies, result dimensions, subject information, etc...)
            which can be completely filled with node/channel-dependent data.

        connectivity_results : dict
        -   Dictionary containing node-dependent data for the connectivity
            analysis that will be copied into the topography results.
        """
        connectivity_results = self.connectivity_results_as_dict()
        remove_keys = [
            f"connectivity-{self._con_method}",
            f"connectivity-{self._con_method}_dimensions",
            "freqs",
            "samp_freq",
            "metadata",
            "processing_steps",
            "subject_info",
        ]
        for key in remove_keys:
            del connectivity_results[key]

        topographies_name = f"connectivity-{self._con_method}_topography"
        topography_results_keys = [
            topographies_name,
            "ch_names",
            "ch_types",
            "ch_coords",
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
            "ch_reref_types",
            "ch_node_types",
            *connectivity_results.keys(),
        ]
        topography_results = {
            f"{topographies_name}_dimensions": ["channels", "frequencies"],
            "freqs": self.results[0].freqs,
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }
        for key in topography_results_keys:
            topography_results[key] = []

        return topography_results, connectivity_results

    def get_results(
        self,
        connectivity_dims: Union[list[str], None] = None,
        topography_dims: Union[list[str], None] = None,
    ) -> tuple[NDArray, list[Union[list, None]]]:
        """Gets the connectivity and, if applicable, topography results of the
        analysis.

        PARAMETERS
        ----------
        connectivity_dims : list[str] | None; default None
        -   The dimensions of the connectivity results that will be returned.
        -   If 'None', the current dimensions are used.

        topography_dims : list[str] | None; default None
        -   The dimensions of the topography results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        results : tuple(numpy array, numpy array) | numpy array
        -   The results.
        -   If spatial topographies have been computed, the output
            is a tuple of two numpy arrays containing the connectivity and
            topography results, respectively, otherwise the output is an array
            containing the connectivity results.
        -   The topography results array contains two subarrays containing the
            spatial weights of the seeds and targets, respectively.
        """
        connectivity_results = self.get_connectivity_results(
            dimensions=connectivity_dims
        )
        if self._return_topographies:
            topography_results = self.get_topography_results(
                dimensions=topography_dims
            )
            results = (connectivity_results, topography_results)
        else:
            results = connectivity_results

        return results

    def get_connectivity_results(
        self, dimensions: Union[list[str], None] = None
    ) -> NDArray:
        """Gets the results of the connectivity analysis.

        PARAMETERS
        ----------
        dimensions : list[str] | None; default None
        -   The dimensions of the results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        numpy array
        -   The connectivity results.
        """
        return super().get_results(dimensions=dimensions)

    def get_topography_results(
        self, dimensions: Union[list[str], None] = None
    ) -> NDArray:
        """Gets the topography results.

        PARAMETERS
        ----------
        dimensions : list[str] | None;  default None
        -   The dimensions of the results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        topographies : numpy array
        -   The topographies of the connectivity results as an array with two
            subarrays containing the topographies for the seed and target
            channels, respectively, for each connectivity node are returned, in
            which case the topographies have dimensions [windows x nodes x
            channels x frequencies] if windows have not been averaged over, or
            [nodes x channels x frequencies] if they have.

        RAISES
        ------
        AttributeError
        -   Raised if spatial topographies have not been computed.
        """
        if not self._return_topographies:
            raise AttributeError(
                "Spatial topography values have not been computed, and so "
                "cannot be returned."
            )

        topographies = []
        if self._windows_averaged:
            topographies.append(self.seed_topographies[0])
            topographies.append(self.target_topographies[0])
        else:
            topographies.append(self.seed_topographies)
            topographies.append(self.target_topographies)

        if dimensions is not None:
            topographies = [
                rearrange_axes(
                    topographies[0], self._topography_results_dims, dimensions
                ),
                rearrange_axes(
                    topographies[1], self._topography_results_dims, dimensions
                ),
            ]

        return topographies


class ConnectivityGranger(ProcMultivariateConnectivity):
    """Calculates multivariate spectral Granger causality between signals.

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
        self._ensure_full_rank_data = None
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
        ensure_full_rank_data: bool = True,
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
        """Performs the Granger casuality (GC) analysis on the data.

        PARAMETERS
        ----------
        gc_method : str
        -   The GC metric to compute.
        -   Supported inputs are: "gc" for GC from seeds to targets; "net_gc"
            for net GC, i.e. GC from seeds to targets minus GC from targets to
            seeds; "trgc" for time-reversed GC (TRGC) from seeds to targets; and
            "net_trgc" for net TRGC, i.e. TRGC from seeds to targets minus TRGC
            from targets to seeds.

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

        average_windows : bool; default True
        -   Whether or not to average connectivity results across windows.

        ensure_full_rank_data : bool; default True
        -   Whether or not to make sure that the data being processed has full
            rank by performing a singular value decomposition on the data of the
            seeds and targets and taking only the first n components, where n is
            equal to number of non-zero singluar values in the decomposition
            (i.e. the rank of the data).
        -   If this is not performed, errors can arise when computing Granger
            causality as assumptions of the method are violated.

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
        self._ensure_full_rank_data = ensure_full_rank_data
        self._n_jobs = n_jobs

        self._sort_processing_inputs()
        self._get_results()
        self._processed = True

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
        self._sort_used_settings()

    def _sort_used_settings(self) -> None:
        """Collects the settings that are relevant for the processing being
        performed and adds only these settings to the 'processing_steps'
        dictionary."""
        used_settings = {
            "granger_causality_method": self._gc_method,
            "cross-spectra_method": self._cs_method,
            "n_lags": self._n_lags,
            "average_windows": self._average_windows,
            "ensure_full_rank_data": self._ensure_full_rank_data,
            "t_min": self._tmin,
            "t_max": self._tmax,
        }

        if self._cs_method == "fourier":
            add_settings = {
                "fmin": self._fmt_fmin,
                "fmax": self._fmt_fmax,
                "n_fft": self._fmt_n_fft,
            }
        elif self._cs_method == "multitaper":
            add_settings = {
                "fmin": self._fmt_fmin,
                "fmax": self._fmt_fmax,
                "n_fft": self._fmt_n_fft,
                "bandwidth": self._mt_bandwidth,
                "adaptive": self._mt_adaptive,
                "low_bias": self._mt_low_bias,
            }
        elif self._cs_method == "cwt_morlet":
            add_settings = {
                "n_cycles": self._cwt_n_cycles,
                "use_fft": self._cwt_use_fft,
                "decim": self._cwt_decim,
            }
        used_settings.update(add_settings)

        self.processing_steps["connectivity_granger"] = used_settings

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""
        if self._verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data) * len(self._indices),
                title="Computing connectivity",
            )

        first_node = True
        for seeds, targets in zip(self._seeds_list, self._targets_list):
            if self._verbose:
                print(
                    f"Computing connectivity.\n- Seeds: {seeds}\n- Targets: "
                    f"{targets}\n"
                )
            seed_data = self._extract_data(seeds)
            target_data = self._extract_data(targets)
            if self._ensure_full_rank_data:
                if first_node:
                    self._seed_ranks = []
                    self._target_ranks = []
                (
                    seed_data,
                    seed_rank,
                    target_data,
                    target_rank,
                ) = self._check_data_rank(
                    seed_data=seed_data, target_data=target_data
                )
            data = self._join_seed_target_data(seed_data, target_data)
            result = self._compute_gc(
                cross_spectra=self._compute_csd(data),
                seeds=np.arange(seed_rank).tolist(),
                targets=np.arange(seed_rank, seed_rank + target_rank).tolist(),
            )
            if first_node:
                results = result.copy()
            else:
                results = np.concatenate((results, result), axis=1)
            first_node = False
        self.results = granger_causality

        if self._progress_bar is not None:
            self._progress_bar.close()

        self._sort_dimensions()
        super()._generate_extra_info()

    def _check_data_rank(
        self, seed_data: NDArray, target_data: NDArray
    ) -> tuple[NDArray, int, NDArray, int]:
        """Checks whether the seed and target data for a node has full rank,
        performing a singular value decomposition (SVD) on the data if not and
        returning only the number of components equal to the number of non-zero
        singular values of the data.

        PARAMETERS
        ----------
        seed_data : numpy ndarray
        -   A 4D matrix of the seed data with dimensions [windows x epochs x
            channels x timepoints].

        target_data : numpy ndarray
        -   A 4D matrix of the target data with dimensions [windows x epochs x
            channels x timepoints].

        RETURNS
        -------
        seed_data : numpy ndarray
        -   A 4D matrix of the seed data with dimensions [windows x epochs x
            rank x timepoints]. If the data has full rank, the data is not
            altered, else data with full rank is returned.

        target_data : numpy ndarray
        -   A 4D matrix of the target data with dimensions [windows x epochs x
            rank x timepoints]. If the data has full rank, the data is not
            altered, else data with full rank is returned.

        seed_rank : int
        -   The rank of the seed data.

        target_rank : int
        -   The rank of the target data.
        """
        seed_data, seed_rank, _ = self._sort_data_dimensionality(
            data=seed_data, data_type="seed"
        )
        target_data, target_rank, _ = self._sort_data_dimensionality(
            data=target_data, data_type="target"
        )
        self._seed_ranks.append(seed_rank)
        self._target_ranks.append(target_rank)

        return seed_data, target_data, seed_rank, target_rank

    def _compute_csd(self, data: list[NDArray]) -> list[CrossSpectralDensity]:
        """Computes the cross-spectral density of the data for a single node.

        PARAMETERS
        ----------
        data : list[numpy ndarray]
        -   The data of a single seed-target pair consisting of a set of 3D
            matrices with dimensions [epochs x channels x timepoints] in a list
            containing the data for individual windows of data.
        -   Channels corresponding to the seeds should be in indices [0 :
            n_seeds], and the targets in indices [n_seeds : end].

        RETURNS
        -------
        cross_spectra : list[MNE CrossSpectralDensity]
        -   The cross-spectra for each window of the data.
        """
        cross_spectra = []
        for win_i, win_data in enumerate(data):
            if self._verbose:
                print(
                    "Computing the cross-spectral density for window "
                    f"{win_i+1} of {len(data)}.\n"
                )
            if self._cs_method == "fourier":
                cross_spectra.append(
                    csd_array_fourier(
                        X=win_data,
                        sfreq=self.signal.data[0].info["sfreq"],
                        t0=0,
                        fmin=self._fmt_fmin,
                        fmax=self._fmt_fmax,
                        tmin=self._tmin,
                        tmax=self._tmax,
                        ch_names=None,
                        n_fft=self._fmt_n_fft,
                        projs=None,
                        n_jobs=self._n_jobs,
                        verbose=self._verbose,
                    )
                )
            elif self._cs_method == "multitaper":
                cross_spectra.append(
                    csd_array_multitaper(
                        X=win_data,
                        sfreq=self.signal.data[0].info["sfreq"],
                        t0=0,
                        fmin=self._fmt_fmin,
                        fmax=self._fmt_fmax,
                        tmin=self._tmin,
                        tmax=self._tmax,
                        ch_names=None,
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
                    csd_array_morlet(
                        X=win_data,
                        sfreq=self.signal.data[0].info["sfreq"],
                        frequencies=self._cwt_freqs,
                        t0=0,
                        tmin=self._tmin,
                        tmax=self._tmax,
                        ch_names=None,
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

    def _compute_gc(
        self,
        cross_spectra: list[CrossSpectralDensity],
        seeds: list[int],
        targets: list[int],
    ) -> NDArray:
        """Computes Granger casuality between seeds and targets of a single node
        for each window.

        PARAMETERS
        ----------
        cross_spectra : list[MNE CrossSpectralDensity]
        -   The cross-spectra between seeds and targets of a single node for
            each window, with seeds in indices [0 : n_seeds] and targets in
            indices [n_seeds : end], for axes 0 and 1.

        seeds : list[int]
        -   Indices of seeds in the cross-spectra in axes 0 and 1.

        targets : list[int]
        -   Indices of targets in the cross-spectra in axes 0 and 1.

        RETURNS
        -------
        results : numpy ndarray
        -   The Granger causality results with dimensions [windows x nodes x
            frequencies].
        """
        results = []
        for i, csd in enumerate(cross_spectra):
            if self._verbose:
                print(
                    f"Computing Granger causality for window {i+1} of "
                    f"{len(cross_spectra)}.\n"
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
                    seeds=[seeds],
                    targets=[targets],
                    n_lags=self._n_lags,
                )
            )
            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        return np.asarray(results)

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
                n_epochs_used=self.signal.data[0].get_data().shape[0],
            )
        ]

        self._windows_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        ask_before_overwrite: Union[bool, None] = None,
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
        results_dict : dict
        -   The results and additional information stored as a dictionary.
        """
        dimensions = self._get_optimal_dims()
        results = super().get_results(dimensions=dimensions)

        results_dict = {
            f"connectivity-{self._gc_method}": results.tolist(),
            f"connectivity-{self._gc_method}_dimensions": dimensions,
            "freqs": self._freqs,
            "seed_names": self._seeds_str,
            "seed_ranks": self._seed_ranks,
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_subregions": self.extra_info["node_ch_subregions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_ch_reref_types"][0],
            "target_names": self._targets_str,
            "target_ranks": self._target_ranks,
            "target_types": self.extra_info["node_ch_types"][1],
            "target_coords": self.extra_info["node_ch_coords"][1],
            "target_regions": self.extra_info["node_ch_regions"][1],
            "target_subregions": self.extra_info["node_ch_subregions"][1],
            "target_hemispheres": self.extra_info["node_ch_hemispheres"][1],
            "target_reref_types": self.extra_info["node_ch_reref_types"][1],
            "node_lateralisation": self.extra_info["node_lateralisation"],
            "node_epoch_orders": self.extra_info["node_ch_epoch_orders"],
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }
        if not self._ensure_full_rank_data:
            remove_keys = ["seed_ranks", "target_ranks"]
            for key in remove_keys:
                del results_dict[key]
