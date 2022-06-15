"""Classes for performing power analysis on data.

CLASSES
-------
PowerMorlet
-   Performs power analysis on preprocessed data using Morlet wavelets.
"""

from copy import deepcopy
from typing import Optional, Union
from matplotlib import pyplot as plt
from fooof import FOOOF
from mne import time_frequency
from numpy.typing import NDArray
import numpy as np
import coh_signal
from coh_exceptions import (
    EntryLengthError,
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_handle_entries import (
    check_lengths_list_identical,
    check_matching_entries,
    ordered_list_from_dict,
    rearrange_axes,
)
from coh_normalisation import norm_percentage_total
from coh_processing_methods import ProcMethod
from coh_saving import save_dict, save_object


class PowerMorlet(ProcMethod):
    """Performs power analysis on data using Morlet wavelets.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   A preprocessed Signal object whose data will be processed.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs Morlet wavelet power analysis using the implementation in
        mne.time_frequency.tfr_morlet.

    save_object
    -   Saves the PowerMorlet object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the power results and additional information as a dictionary.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal=signal, verbose=verbose)

        # Initialises inputs of the PowerMorlet object.
        self._sort_inputs()

        # Initialises aspects of the PowerMorlet object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._epochs_averaged = False
        self._timepoints_averaged = False
        self._normalised = False

    def _sort_inputs(self) -> None:
        """Checks the inputs to the PowerMorlet object to ensure that they
        match the requirements for processing and assigns inputs.

        RAISES
        ------
        InputTypeError
        -   Raised if the Signal object input does not contain epoched data.
        """

        if "epochs" not in self.signal.data_dimensions:
            raise TypeError(
                "The provided Signal object does not contain epoched data. "
                "Epoched data is required for power analysis."
            )

        super()._sort_inputs()

    def _average_windows(self) -> None:
        """Averages the power results across windows.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the windows have already been averaged across.
        """

        if self._windows_averaged:
            raise ProcessingOrderError(
                "Trying to average the results across windows, but this has "
                "already been done."
            )

        n_windows = len(self.results)
        power = []
        for results in self.results:
            power.append(results.data)
        self.results[0].data = np.asarray(power).mean(axis=0)
        self.results = [self.results[0]]

        self._windows_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

    def _average_timepoints(self) -> None:
        """Averages results of the analysis across timepoints.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the timepoints have already been averaged across.
        """

        if self._timepoints_averaged:
            raise ProcessingOrderError(
                "Trying to average the results across timepoints, but this has "
                "already been done."
            )

        timepoints_i = self.results_dims.index("timepoints")
        n_timepoints = np.shape(self.results[0].data)[timepoints_i]
        for i, power in enumerate(self.results):
            self.results[i].data = np.mean(power.data, axis=timepoints_i)
        self._results_dims.pop(timepoints_i + 1)

        self._timepoints_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_timepoints} timepoints.")

    def process(
        self,
        freqs: list[Union[int, float]],
        n_cycles: Union[int, list[int]],
        use_fft: bool = False,
        decim: Union[int, slice] = 1,
        n_jobs: int = 1,
        picks: Optional[list[int]] = None,
        zero_mean: bool = True,
        average_windows: bool = False,
        average_epochs: bool = False,
        average_timepoints: bool = False,
        output: str = "power",
    ) -> None:
        """Performs Morlet wavelet power analysis using the implementation in
        mne.time_frequency.tfr_morlet.

        PARAMETERS
        ----------
        freqs : list[realnum]
        -   The frequencies in Hz to analyse.

        n_cycles : int | list[int]
        -   The number of cycles globally (if int) or for each frequency (if
            list[int]).

        use_fft : bool; default False
        -   Whether or not to perform the fft based convolution.

        decim : int | slice; default 1
        -   Decimates data following time-frequency decomposition. Returns
            data[..., ::decim] if int. Returns data[..., decim]. Warning: may
            create decimation artefacts.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel on the CPU. If -1, it is set
            to the number of CPU cores. Requires the joblib package.

        picks : list[int] | None; default None
        -   The indices of the channels to decompose. If None, all good data
            channels are decomposed.

        zero_mean : bool; default True
        -   Gives the wavelet a mean of 0.

        average_windows : bool; default False
        -   If True, averages the power across windows. If False, returns
            separate power values for each window.

        average_epochs : bool; default False
        -   If True, averages the power across epochs. If False, returns
            separate power values for each epoch.

        average_timepoints : bool; default False
        -   If True, averages the power across timepoints within each epoch. If
            False, returns separate power values for each timepoint in the
            epoch.

        output : str; default 'power'
        -   Can be 'power' or 'complex'. If 'complex', average_epochs must be
            False.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data in the object has already been processed.
        """

        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        if self._verbose:
            print("Performing Morlet wavelet power analysis on the data.\n")

        self.results = self._get_results(
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=use_fft,
            decim=decim,
            n_jobs=n_jobs,
            picks=picks,
            zero_mean=zero_mean,
            average_epochs=average_epochs,
            output=output,
        )

        self._epochs_averaged = average_epochs
        self._establish_power_dims()
        if average_windows:
            self._average_windows()
        if average_timepoints:
            self._average_timepoints()

        self._processed = True
        self.processing_steps["power_morlet"] = {
            "freqs": freqs,
            "n_cycles": n_cycles,
            "use_fft": use_fft,
            "decim": decim,
            "n_jobs": n_jobs,
            "picks": picks,
            "zero_mean": zero_mean,
            "average_windows": average_windows,
            "average_epochs": average_epochs,
            "average_timepoints": average_timepoints,
            "output": output,
        }

    def _get_results(
        self,
        freqs: list[Union[int, float]],
        n_cycles: Union[int, list[int]],
        use_fft: bool,
        decim: Union[int, slice],
        n_jobs: int,
        picks: list[int],
        zero_mean: bool,
        average_epochs: bool,
        output: str,
    ) -> list:
        """"""

        results = []
        for i, data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"\n---=== Computing power for window {i+1} of "
                    f"{len(self.signal.data)} ===---\n"
                )
            results.append(
                time_frequency.tfr_morlet(
                    inst=data,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=use_fft,
                    return_itc=False,
                    decim=decim,
                    n_jobs=n_jobs,
                    picks=picks,
                    zero_mean=zero_mean,
                    average=average_epochs,
                    output=output,
                    verbose=self._verbose,
                )
            )

        return results

    def _establish_power_dims(self) -> None:
        """Establishes the dimensions of the power results."""

        self._results_dims = [
            "windows",
            "epochs",
            "channels",
            "frequencies",
            "timepoints",
        ]
        if self._epochs_averaged:
            self._results_dims.pop(self._results_dims.index("epochs"))

    def _sort_normalisation_inputs(
        self,
        norm_type: str,
        within_dim: str,
        exclude_line_noise_window: Union[int, float, None] = None,
        line_noise_freq: Union[int, float, None] = None,
    ) -> None:
        """Sorts the user inputs for the normalisation of results.

        PARAMETERS
        ----------
        norm_type : str
        -   The type of normalisation to apply.
        -   Currently, only "percentage_total" is supported.

        within_dim : str
        -   The dimension to apply the normalisation within.

        exclusion_line_noise_window : int | float | None; default None
        -   The size of the windows (in Hz) to exclude frequencies around the
            line noise and harmonic frequencies from the calculations of what to
            normalise the data by.

        line_noise_freq : int | float | None; default None
        -   Frequency (in Hz) of the line noise.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if the requested normalisation type is not supported.
        -   Raised if the length of the results dimensions are greater than the
            maximum number supported.
        -   Raised if the dimension to normalise across is not present in the
            results dimensions.

        ValueError
        -   Raised if a window of results are to be excluded from the
            normalisation around the line noise, but no line noise frequency is
            given.
        """

        supported_norm_types = ["percentage_total"]
        max_supported_n_dims = 2

        if norm_type not in supported_norm_types:
            raise UnavailableProcessingError(
                "Error when normalising the results of the Morlet power "
                f"analysis:\nThe normalisation type '{norm_type}' is not "
                "supported. The supported normalisation types are: "
                f"{supported_norm_types}."
            )

        if len(self.results_dims) > max_supported_n_dims:
            raise UnavailableProcessingError(
                "Error when normalising the results of the Morlet power "
                "analysis:\nCurrently, normalising the values of results with "
                f"at most {max_supported_n_dims} is supported, but the results "
                f"have {len(self.results_dims)} dimensions."
            )

        if within_dim not in self.results_dims:
            raise UnavailableProcessingError(
                "Error when normalising the results of the Morlet power "
                f"analysis:\nThe dimension '{within_dim}' is not present in "
                f"the results of dimensions {self.results_dims}."
            )

        if line_noise_freq is None and exclude_line_noise_window is not None:
            raise ValueError(
                "Error when normalising the results of the Morlet power "
                "analysis:\nYou have requested a window to be excluded around "
                "the line noise, but no line noise frequency has been "
                "provided."
            )

    def normalise(
        self,
        norm_type: str,
        within_dim: str,
        exclude_line_noise_window: Union[int, float, None] = None,
        line_noise_freq: Union[int, float, None] = None,
    ) -> None:
        """Normalises the results of the Morlet power analysis.

        PARAMETERS
        ----------
        norm_type : str
        -   The type of normalisation to apply.
        -   Currently, only "percentage_total" is supported.

        within_dim : str
        -   The dimension to apply the normalisation within.
        -   E.g. if the data has dimensions "channels" and "frequencies",
            setting 'within_dims' to "channels" would normalise the data across
            the frequencies within each channel.
        -   Currently, normalising only two-dimensional data is supported.

        exclusion_line_noise_window : int | float | None; default None
        -   The size of the windows (in Hz) to exclude frequencies around the
            line noise and harmonic frequencies from the calculations of what to
            normalise the data by.
        -   If None, no frequencies are excluded.
        -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is
            10, the results from 45 - 55 Hz would be ommited.

        line_noise_freq : int | float | None; default None
        -   Frequency (in Hz) of the line noise.
        -   If None, 'exclusion_line_noise_window' must also be None.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the results have already been normalised.
        """

        if self._normalised:
            raise ProcessingOrderError(
                "Error when normalising the results of the Morlet power "
                "analysis:\nThe results have already been normalised."
            )

        self._sort_normalisation_inputs(
            norm_type=norm_type,
            within_dim=within_dim,
            exclude_line_noise_window=exclude_line_noise_window,
            line_noise_freq=line_noise_freq,
        )

        if norm_type == "percentage_total":
            for i, power in enumerate(self.results):
                self.results[i].data = norm_percentage_total(
                    data=deepcopy(power.data),
                    freqs=self.results[0].freqs,
                    data_dims=self._results_dims[1:],
                    within_dim=within_dim,
                    line_noise_freq=line_noise_freq,
                    exclusion_window=exclude_line_noise_window,
                )

        self._normalised = True
        self.processing_steps["power_morlet_normalisation"] = {
            "normalisation_type": norm_type,
            "within_dim": within_dim,
            "exclude_line_noise_window": exclude_line_noise_window,
            "line_noise_freq": line_noise_freq,
        }
        if self._verbose:
            print(
                f"Normalising results with type '{norm_type}' on the "
                f"{within_dim} dimension using a line noise exclusion window "
                f"at {line_noise_freq} of {exclude_line_noise_window} Hz."
            )

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the PowerMorlet object as a .pkl file.

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

    def results_as_dict(self) -> dict:
        """Returns the power results and additional information as a dictionary.

        RETURNS
        -------
        dict
        -   The results and additional information stored as a dictionary.
        """

        dimensions = self._get_optimal_dims()
        results = self.get_results(dimensions=dimensions)

        results = {
            "power-morlet": results.tolist(),
            "power-morlet_dimensions": dimensions,
            "freqs": self.results[0].freqs.tolist(),
            "ch_names": self.results[0].ch_names,
            "ch_types": self.results[0].get_channel_types(),
            "ch_coords": self.signal.get_coordinates(),
            "ch_regions": ordered_list_from_dict(
                self.results[0].ch_names, self.extra_info["ch_regions"]
            ),
            "ch_hemispheres": ordered_list_from_dict(
                self.results[0].ch_names, self.extra_info["ch_hemispheres"]
            ),
            "ch_reref_types": ordered_list_from_dict(
                self.results[0].ch_names, self.extra_info["ch_reref_types"]
            ),
            "samp_freq": self.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }

        return results

    def get_results(self, dimensions: Union[list[str], None] = None) -> NDArray:
        """Extracts and returns results.

        RETURNS
        -------
        results : numpy array
        -   The results.
        """

        ch_names = [self.signal.data[0].ch_names]
        ch_names.extend([results.ch_names for results in self.results])
        if not check_matching_entries(objects=ch_names):
            raise ValueError(
                "Error when getting the results:\nThe names and/or order of "
                "names in the data and results do not match."
            )

        if dimensions is None:
            dimensions = self.results_dims

        if self._windows_averaged:
            results = self.results[0].data
        else:
            results = []
            for mne_obj in self.results:
                results.append(mne_obj.data)
            results = np.asarray(results)

        results = rearrange_axes(
            obj=results, old_order=self.results_dims, new_order=dimensions
        )

        return deepcopy(results)

    def save_results(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the power results and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the power results should be saved.

        ftype : str | None; default None
        -   The filetype of the power results that will be saved, without the
            leading period. E.g. for saving the file in the json format, this
            would be "json", not ".json".
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


class PowerFOOOF(ProcMethod):
    """Performs power analysis on data using FOOOF.

    PARAMETERS
    ----------
    signal : PowerMorlet
    -   Power spectra data that will be processed using FOOOF.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs FOOOF power analysis.

    save_object
    -   Saves the PowerFOOOF object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the FOOOF analysis results and additional information as a
        dictionary.
    """

    def __init__(self, signal: PowerMorlet, verbose: bool = True) -> None:
        super().__init__(signal=signal, verbose=verbose)

        # Initialises inputs of the PowerFOOOF object.
        self._sort_inputs()

        # Initialises aspects of the PowerFOOOF object that will be filled with
        # information as the data is processed.
        self.freq_range = None
        self.aperiodic_modes = None
        self.peak_width_limits = None
        self.max_n_peaks = None
        self.min_peak_height = None
        self.peak_threshold = None
        self.show_fit = None
        self.fooof_results = None

    def _sort_inputs(self) -> None:
        """Checks the inputs to the PowerFOOOF object to ensure that they
        match the requirements for processing and assigns inputs.

        RAISES
        ------
        InputTypeError
        -   Raised if the PowerMorlet object input does not contain data in a
            supported format.
        """

        supported_data_dims = [["channels", "frequencies"]]
        if self.signal.results_dims not in supported_data_dims:
            raise TypeError(
                "Error when applying FOOOF to the power data:\nThe data in the "
                f"power object is in the form {self.signal.results_dims}, but "
                f"only data in the form {supported_data_dims} is supported."
            )
        if self.signal._windows_averaged:
            self.signal.results = self.signal.results[0]

        super()._sort_inputs()

    def _sort_processing_inputs(
        self,
        freq_range: Union[list[int, float], None],
        peak_width_limits: list[Union[int, float]],
        max_n_peaks: Union[int, float],
        min_peak_height: Union[int, float],
        peak_threshold: Union[int, float],
        aperiodic_modes: Union[list[Union[str, None]], None],
        show_fit: bool,
    ) -> None:
        """Sorts and assigns inputs for the FOOOF processing.

        PARAMETERS
        ----------
        freq_range : list[int | float] | None
        -   The lower and upper frequency limits, respectively, in Hz, for
            which the FOOOF analysis should be performed.
        -   If 'None', the entire frequency range of the power spectra is used.

        peak_width_limits : list[int | float]
        -   Minimum and maximum limits, respectively, in Hz, for detecting peaks
            in the data.

        max_n_peaks : int | inf
        -   The maximum number of peaks that will be fit.

        min_peak_height : int | float
        -   Minimum threshold, in units of the input data, for detecing peaks.

        peak_threshold : int | float
        -   Relative threshold, in units of standard deviation of the input data
            for detecting peaks.

        aperiodic_modes : list[str | None] | None
        -   The mode for fitting the periodic component, can be "fixed" or
            "knee".
        -   If 'None', the user is shown, for each channel individually, the
            results of the different fits and asked to choose the fits to use
            for each channel.
        -   If some entries are 'None', the user is show the results of the
            different fits for these channels and asked to choose the fits to
            use for each of these channels.

        show_fit : bool
        -   Whether or not to show the user the FOOOF model fits for the
            channels.
        """

        if freq_range is None:
            freq_range = [None, None]
            freq_range[0] = self.signal.results.freqs[0]
            freq_range[1] = self.signal.results.freqs[-1]
        self.freq_range = freq_range

        if aperiodic_modes is None:
            aperiodic_modes = [None] * len(self.signal.results.ch_names)
        else:
            identical, lengths = check_lengths_list_identical(
                to_check=[self.signal.results.ch_names, aperiodic_modes]
            )
            if not identical:
                raise EntryLengthError(
                    "Error when applying FOOOF to the power data:\nThe "
                    f"requested {lengths[1]} aperiodic fitting modes do not "
                    f"match the number of channels in the data {lengths[0]}."
                )
        self.aperiodic_modes = aperiodic_modes

        self.peak_width_limits = peak_width_limits
        self.max_n_peaks = max_n_peaks
        self.min_peak_height = min_peak_height
        self.peak_threshold = peak_threshold
        self.show_fit = show_fit

    def _fit_fooof_model(
        self, channel_index: int, aperiodic_mode: Optional[str] = None
    ) -> FOOOF:
        """Fits a FOOOF model to the power data of a single channel according to
        the pre-specified settings.

        PARAMETERS
        ----------
        channel_index : int
        -   The index of the channel in the data to apply the FOOOF model to.

        aperiodic_mode : str | None; default None
        -   The aperiodic mode to use when fitting the model.
        -   Recognised modes are 'fixed' and 'knee'.
        -   If None, the mode is chosen based on the already-assigned mode for
            this channel.

        RETURNS
        -------
        fooof_model : fooof FOOOF
        -   The fitted FOOOF model.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if the aperiodic mode is not given and the pre-specified
            aperiodic mode for this channel is 'None'.
        """

        if aperiodic_mode is None:
            aperiodic_mode = self.aperiodic_modes[channel_index]
            report_fit = True
        else:
            report_fit = False
        if aperiodic_mode is None:
            raise UnavailableProcessingError(
                "Error when fitting the FOOOF model:\nThe aperiodic mode is "
                "set to 'None', however FOOOF recognises this as the default "
                "choice of a 'fixed' mode. When fitting a model, the aperiodic "
                "mode must be specified."
            )

        if self._verbose and report_fit:
            print(
                "Fitting the FOOOF model for the data of channel "
                f"'{self.signal.results.ch_names[channel_index]}' with aperiodic "
                f"mode '{aperiodic_mode}'."
            )

        fooof_model = FOOOF(
            peak_width_limits=self.peak_width_limits,
            max_n_peaks=self.max_n_peaks,
            min_peak_height=self.min_peak_height,
            peak_threshold=self.peak_threshold,
            aperiodic_mode=aperiodic_mode,
            verbose=self._verbose,
        )
        fooof_model.fit(
            freqs=self.signal.results.freqs,
            power_spectrum=self.signal.results.data[channel_index],
            freq_range=self.freq_range,
        )

        return fooof_model

    def _plot_aperiodic_modes(
        self, fm_fixed: FOOOF, fm_knee: FOOOF, channel_index: int
    ) -> None:
        """Plots the fitted FOOOF models using the 'fixed' and 'knee' aperiodic
        modes together alongside goodness of fit metrics in log-standard and
        log-log forms.

        PARAMETERS
        ----------
        fm_fixed : fooof FOOOF
        -   The fitted FOOOF model with a fixed aperiodic component.

        fm_knee : fooof FOOOF
        -   The fitted FOOOF model with a knee aperiodic component.

        channel_index : int
        -   The index of the channel whose data is being plotted.
        """

        fig, axs = plt.subplots(2, 2)
        fig.suptitle(self.signal.results.ch_names[channel_index])

        fm_fixed.plot(plt_log=False, ax=axs[0, 0])
        fm_fixed.plot(plt_log=True, ax=axs[1, 0])

        fm_knee.plot(plt_log=False, ax=axs[0, 1])
        fm_knee.plot(plt_log=True, ax=axs[1, 1])

        fit_metrics = ["r_squared_", "error_"]
        fixed_mode_title = "Fixed aperiodic mode | "
        knee_mode_title = "Knee aperiodic mode | "
        for metric in fit_metrics:
            fixed_mode_title += (
                f"{metric}={round(getattr(fm_fixed, metric), 2)} | "
            )
            knee_mode_title += (
                f"{metric}={round(getattr(fm_knee, metric), 2)} | "
            )
        axs[0, 0].set_title(fixed_mode_title)
        axs[0, 1].set_title(knee_mode_title)

        plt.show()

    def _input_aperiodic_mode_choice(self) -> str:
        """Asks the user to choose which aperiodic mode should be used to fit
        the FOOOF model.

        RETURNS
        -------
        aperiodic_mode : str
        -   The chosen aperiodic mode.
        -   Can be 'fixed' or 'knee'.
        """

        answered = False
        accepted_inputs = ["fixed", "knee"]

        while not answered:
            aperiodic_mode = input(
                "Which mode should be used for fitting the aperiodic "
                "component, 'fixed' or 'knee'?: "
            )
            if aperiodic_mode in accepted_inputs:
                answered = True
            else:
                print(
                    f"Error, {aperiodic_mode} is not a recognised input.\nThe "
                    f"recognised inputs are {accepted_inputs}.\nPlease try "
                    "again."
                )

        return aperiodic_mode

    def _choose_aperiodic_mode(self, channel_index: int) -> str:
        """Fits the FOOOF model to the data for both the 'fixed' and 'knee'
        aperiodic modes, and asks the user to choose the desired mode.

        PARAMETERS
        ----------
        channel_index : int
        -   The index of the channel in the data which should be examined.

        RETURNS
        -------
        aperiodic_mode : str
        -   The aperiodic mode chosen by the user.
        -   Will be 'fixed' or 'knee'.
        """

        if self._verbose:
            print(
                "No aperiodic mode has been specified for channel "
                f"'{self.signal.results.ch_names[channel_index]}'. Please "
                "choose an aperiodic mode to use after closing the figure."
            )

        fm_fixed = self._fit_fooof_model(
            channel_index=channel_index, aperiodic_mode="fixed"
        )
        fm_knee = self._fit_fooof_model(
            channel_index=channel_index, aperiodic_mode="knee"
        )

        self._plot_aperiodic_modes(
            fm_fixed=fm_fixed, fm_knee=fm_knee, channel_index=channel_index
        )

        self.aperiodic_modes[
            channel_index
        ] = self._input_aperiodic_mode_choice()

    def _inspect_model(self, model: FOOOF, channel_index: int) -> None:
        """Plots the fitted FOOOF model alongside the goodness-of-fit metrics.

        PARAMETERS
        ----------
        model : FOOOF
        -   The fitted FOOOF model to inspect.

        channel_index : int
        -   The index of the channel whose data is included in the model.
        """

        fig, axs = plt.subplots(2, 1)
        model_title = (
            f"{self.signal.results.ch_names[channel_index]} | "
            f"{self.aperiodic_modes[channel_index]} aperiodic mode | "
        )

        model.plot(plt_log=False, ax=axs[0])
        model.plot(plt_log=True, ax=axs[1])

        fit_metrics = ["r_squared_", "error_"]
        for metric in fit_metrics:
            model_title += f"{metric}={round(getattr(model, metric), 2)} | "
        fig.suptitle(model_title)

        plt.show()

        answered = False
        while not answered:
            response = input("Press enter to continue...")
            if response:
                answered = True

    def _extract_results_from_model(self, model: FOOOF) -> dict:
        """Extracts the results of the FOOOF model to a dictionary.

        PARAMETERS
        ----------
        model : FOOOF
        -   The FOOOF model fitted to the data.

        RETURNS
        -------
        dict
        -   The results of the FOOOF analysis.
        """

        return {
            "periodic_component": model._spectrum_flat,
            "aperiodic_component": model._spectrum_peak_rm,
            "r_squared": model.get_results().r_squared,
            "error": model.get_results().error,
            "aperiodic_params": model.get_results().aperiodic_params,
            "peak_params": model.get_results().peak_params,
            "gaussian_params": model.get_results().gaussian_params,
        }

    def _apply_fooof(self, channel_index: int) -> dict:
        """Fits a FOOOF model to a single channel of data based on the
        pre-specified settings and organises the results.

        PARAMETERS
        ----------
        channel_index : int
        -   The index of the channel whose data is being analysed.

        RETURNS
        -------
        dict
        -   The results of the FOOOF analysis.
        """

        fooof_model = self._fit_fooof_model(channel_index=channel_index)

        if self.show_fit:
            self._inspect_model(model=fooof_model, channel_index=channel_index)

        return self._extract_results_from_model(model=fooof_model)

    def _combine_results_over_channels(
        self, results: dict[list]
    ) -> dict[np.array]:
        """Converts the entries of the results from lists in which each entry
        represents the values for a single channel into arrays containing the
        values of all channels.

        PARAMETERS
        ----------
        results : dict[list]
        -   The results of all channels stored as lists of arrays, where each
            array within a list correpsonds to the data of a single channel.

        RETURNS
        -------
        results : dict[numpy array]
        -   The results of all channels stored as arrays of arrays.
        """

        ragged_entries = ["peak_params", "gaussian_params"]

        for key, value in results.items():
            dtype = None
            if key in ragged_entries:
                dtype = object
            results[key] = np.asarray(value, dtype=dtype)

        return results

    def _get_results(
        self,
    ) -> dict[np.array]:
        """Fits the FOOOF models to the data.

        RETURNS
        -------
        dict[numpy array]
        -   The results of the FOOOF analysis of all channels.
        """

        results = {
            "periodic_component": [],
            "aperiodic_component": [],
            "r_squared": [],
            "error": [],
            "aperiodic_params": [],
            "peak_params": [],
            "gaussian_params": [],
        }

        for channel_index, aperiodic_mode in enumerate(self.aperiodic_modes):
            if aperiodic_mode is None:
                self._choose_aperiodic_mode(channel_index=channel_index)
            result = self._apply_fooof(channel_index=channel_index)
            for key, item in result.items():
                results[key].append(item)

        return self._combine_results_over_channels(results=results)

    def process(
        self,
        freq_range: Optional[list[int, float]] = None,
        peak_width_limits: list[Union[int, float]] = [0.5, 12],
        max_n_peaks: Union[int, float] = float("inf"),
        min_peak_height: Union[int, float] = 0,
        peak_threshold: Union[int, float] = 2,
        aperiodic_modes: Optional[list[Union[str, None]]] = None,
        show_fit: bool = False,
    ) -> None:
        """Performs FOOOF analysis on the power data.

        PARAMETERS
        ----------
        freq_range : list[int | float] | None; default None
        -   The lower and upper frequency limits, respectively, in Hz, for
            which the FOOOF analysis should be performed.
        -   If 'None', the entire frequency range of the power spectra is used.

        peak_width_limits : list[int | float]; default [0.5, 12]
        -   Minimum and maximum limits, respectively, in Hz, for detecting peaks
            in the data.

        max_n_peaks : int | inf; default inf
        -   The maximum number of peaks that will be fit.

        min_peak_height : int | float; default 0
        -   Minimum threshold, in units of the input data, for detecing peaks.

        peak_threshold : int | float; 2
        -   Relative threshold, in units of standard deviation of the input data
            for detecting peaks.

        aperiodic_modes : list[str | None] | None; default None
        -   The mode for fitting the periodic component, can be "fixed" or
            "knee".
        -   If 'None', the user is shown, for each channel individually, the
            results of the different fits and asked to choose the fits to use
            for each channel.
        -   If some entries are 'None', the user is show the results of the
            different fits for these channels and asked to choose the fits to
            use for each of these channels.

        show_fit : bool; default True
        -   Whether or not to show the user the FOOOF model fits for the
            channels.
        """

        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self._sort_processing_inputs(
            freq_range=freq_range,
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            aperiodic_modes=aperiodic_modes,
            show_fit=show_fit,
        )

        if self._verbose:
            print("Performing FOOOF analysis on the data.\n")

        self.results = self._get_results()

        self._processed = True
        self.processing_steps["power_FOOOF"] = {
            "freq_range": self.freq_range,
            "peak_width_limits": self.peak_width_limits,
            "max_n_peaks": self.max_n_peaks,
            "min_peak_height": self.min_peak_height,
            "peak_threshold": self.peak_threshold,
            "aperiodic_modes": self.aperiodic_modes,
        }

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the PowerMorlet object as a .pkl file.

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

    def get_results(self) -> dict:
        """Returns the results dictionary.

        RETURNS
        -------
        dict
        -   The results dictionary of the FOOOF analysis.
        """

        return deepcopy(self.results)

    def results_as_dict(self) -> dict:
        """Organises the results and additional information into a dictionary.

        RETURNS
        dict
        -   A dictionary of results and additional information.
        """

        return {
            "power-fooof_periodic": self.results["periodic_component"].tolist(),
            "power-fooof_aperiodic": self.results[
                "aperiodic_component"
            ].tolist(),
            "r_squared": self.results["r_squared"].tolist(),
            "error": self.results["error"].tolist(),
            "freqs": self.signal.results.freqs.tolist(),
            "ch_names": self.signal.results.ch_names,
            "ch_types": self.signal.results.get_channel_types(),
            "ch_coords": self.signal.signal.get_coordinates(),
            "ch_regions": ordered_list_from_dict(
                self.signal.results.ch_names, self.extra_info["ch_regions"]
            ),
            "ch_hemispheres": ordered_list_from_dict(
                self.signal.results.ch_names, self.extra_info["ch_hemispheres"]
            ),
            "ch_reref_types": ordered_list_from_dict(
                self.signal.results.ch_names, self.extra_info["ch_reref_types"]
            ),
            "samp_freq": self.signal.results.info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.results.info["subject_info"],
        }

    def save_results(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the FOOOF analysis results and additional information as a
        file.

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

        RAISES
        ------
        EntryLengthError
        -   Raised if the number of channels in the pre-processed data does not
            match the number of channels in the results.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        identical, lengths = check_lengths_list_identical(
            to_check=[
                self.results["periodic_component"],
                self.results["aperiodic_component"],
                self.signal.results.ch_names,
            ]
        )
        if not identical:
            raise EntryLengthError(
                "Error when trying to save the results of the FOOOF power "
                "analysis:\nThe number of channels in the power data "
                f"({lengths[1]}) and in the FOOOF results ({lengths[0]}) do "
                "not match."
            )

        save_dict(
            to_save=self.results_as_dict(),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )
