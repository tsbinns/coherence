"""Classes for performing power analysis on data.

CLASSES
-------
PowerMorlet
-   Performs power analysis on preprocessed data using Morlet wavelets.
"""

from copy import deepcopy
from typing import Union
from unittest.util import sorted_list_difference
from matplotlib import pyplot as plt
from fooof import FOOOF
from fooof.analysis.periodic import get_band_peak
from mne import time_frequency
from numpy.typing import NDArray
import numpy as np
import coh_signal
from coh_exceptions import ProcessingOrderError
from coh_handle_entries import (
    check_lengths_list_identical,
    check_matching_entries,
    ordered_list_from_dict,
    rearrange_axes,
)
from coh_handle_objects import FillableObject
from coh_normalisation import norm_percentage_total
from coh_processing_methods import ProcMethod
from coh_saving import save_dict, save_object


class PowerStandard(ProcMethod):
    """Performs power analysis on data using Welch's method, multitapers, or
    Morlet wavelets.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   A preprocessed Signal object whose data will be processed.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs power analysis on the data.

    save_object
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the power results and additional information as a dictionary.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal=signal, verbose=verbose)

        # Initialises inputs of the PowerMorlet object.
        self._sort_inputs()

        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self._power_method = None

        # Initialises aspects of the PowerMorlet object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._epochs_averaged = False
        self._timepoints_averaged = False
        self._segments_averaged = False
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

    def process_welch(
        self,
        fmin: Union[int, float] = 0,
        fmax: Union[int, float] = np.inf,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        n_fft: int = 256,
        n_overlap: int = 0,
        n_per_seg: Union[int, None] = None,
        proj: bool = False,
        window_method: Union[str, float, tuple] = "hamming",
        average_windows: bool = True,
        average_epochs: bool = True,
        average_segments: Union[str, None] = "mean",
        n_jobs: int = 1,
    ) -> None:
        """Calculates the power spectral density of the data with Welch's
        method, using the implementation in MNE 'time_frequency.psd_welch'.

        Welch's method involves calculating periodograms for a sliding window
        over time which are then averaged together for each channel/epoch.

        PARAMETERS
        ----------
        fmin : int | float; default 0
        -   The minimum frequency of interest.

        fmax : int | float; default infinite
        -   The maximum frequency of interest.

        tmin : float | None; default None
        -   The minimum time of interest.

        tmax : float | None; default None
        -   The maximum time of interest.

        n_fft : int; default 256
        -   The length of the FFT used. Must be >= 'n_per_seg'. If 'n_fft' >
            'n_per_seg', the segments will be zero-padded.

        n_overlap : int; default 0
        -   The number of points to overlap between segments, which will be
            adjusted to be <= the number of time points in the data.

        n_per_seg : int | None; default None
        -   The length of each Welch segment, windowed by a Hamming window. If
            'None', 'n_per_seg' is set to equal 'n_fft', in which case 'n_fft'
            must be <= the number of time points in the data.

        proj : bool; default False
        -   Whether or not to apply SSP projection vectors.

        window_method : str | float | tuple; default "hamming"
        -   The windowing function to use. See scipy's 'signal.get_window'
            function.

        average_windows : bool; default True
        -   Whether or not to average power results across data windows.

        average_epochs : bool; default True
        -   Whether or not to average power results across epochs.

        average_segments : str | None; default "mean"
        -   How to average segments. If "mean", the arithmetic mean is used. If
            "median", the median is used, corrected for bias relative to the
            mean. If 'None', the segments are unagreggated.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel. If '-1', this is set to the
            number of CPU cores. Requires the 'joblib' package.

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
            print("Performing Welch's power analysis on the data.\n")

        self._get_results_welch(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            n_per_seg=n_per_seg,
            proj=proj,
            window_method=window_method,
            average_windows=average_windows,
            average_epochs=average_epochs,
            average_segments=average_segments,
            n_jobs=n_jobs,
        )

        self._processed = True
        self._power_method = "welch"
        self.processing_steps["power_welch"] = {
            "fmin": fmin,
            "fmax": fmax,
            "tmin": tmin,
            "tmax": tmax,
            "n_fft": n_fft,
            "n_overlap": n_overlap,
            "n_per_seg": n_per_seg,
            "proj": proj,
            "window_method": window_method,
            "average_windows": average_windows,
            "average_epochs": average_epochs,
            "average_segments": average_segments,
        }

    def _get_results_welch(
        self,
        fmin: Union[int, float],
        fmax: Union[int, float],
        tmin: Union[float, None],
        tmax: Union[float, None],
        n_fft: int,
        n_overlap: int,
        n_per_seg: Union[int, None],
        proj: bool,
        window_method: Union[str, float, tuple],
        average_windows: bool,
        average_epochs: bool,
        average_segments: Union[str, None],
        n_jobs: int,
    ) -> None:
        """Calculates the power spectral density of the data with Welch's
        method, using the implementation in MNE 'time_frequency.psd_welch'.

        Welch's method involves calculating periodograms for a sliding window
        over time which are then averaged together for each channel/epoch.

        PARAMETERS
        ----------
        fmin : int | float; default 0
        -   The minimum frequency of interest.

        fmax : int | float; default infinite
        -   The maximum frequency of interest.

        tmin : float | None; default None
        -   The minimum time of interest.

        tmax : float | None; default None
        -   The maximum time of interest.

        n_fft : int; default 256
        -   The length of the FFT used. Must be >= 'n_per_seg'. If 'n_fft' >
            'n_per_seg', the segments will be zero-padded.

        n_overlap : int; default 0
        -   The number of points to overlap between segments, which will be
            adjusted to be <= the number of time points in the data.

        n_per_seg : int | None; default None
        -   The length of each Welch segment, windowed by a Hamming window. If
            'None', 'n_per_seg' is set to equal 'n_fft', in which case 'n_fft'
            must be <= the number of time points in the data.

        proj : bool; default False
        -   Whether or not to apply SSP projection vectors.

        window_method : str | float | tuple; default "hamming"
        -   The windowing function to use. See scipy's 'signal.get_window'
            function.

        average_windows : bool; default True
        -   Whether or not to average power results across data windows.

        average_epochs : bool; default True
        -   Whether or not to average power results across epochs.

        average_segments : str | None; default "mean"
        -   How to average segments. If "mean", the arithmetic mean is used. If
            "median", the median is used, corrected for bias relative to the
            mean. If 'None', the segments are unagreggated.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel. If '-1', this is set to the
            number of CPU cores. Requires the 'joblib' package.
        """
        results = []
        ch_names = self.signal.data[0].ch_names
        ch_types = self.signal.data[0].get_channel_types(picks=ch_names)
        for i, data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"\n---=== Computing power for window {i+1} of "
                    f"{len(self.signal.data)} ===---\n"
                )
            psds, freqs = time_frequency.psd_welch(
                inst=data,
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                n_fft=n_fft,
                n_overlap=n_overlap,
                n_per_seg=n_per_seg,
                picks=self.signal.data[0].ch_names,
                proj=proj,
                n_jobs=n_jobs,
                reject_by_annotation=False,
                average=None,
                window=window_method,
                verbose=self._verbose,
            )
            results.append(
                FillableObject(
                    attrs={
                        "data": psds,
                        "freqs": freqs,
                        "ch_names": ch_names,
                        "ch_types": ch_types,
                    }
                )
            )
        self.results = results

        self._results_dims = [
            "windows",
            "epochs",
            "channels",
            "frequencies",
            "segments",
        ]
        if average_windows:
            self._average_results_dim("windows")
        if average_epochs:
            self._average_results_dim("epochs")
        if average_segments:
            self._average_results_dim("segments")

    def process_multitaper(
        self,
        fmin: Union[int, float] = 0,
        fmax: float = np.inf,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        bandwidth: Union[float, None] = None,
        adaptive: bool = False,
        low_bias: bool = True,
        normalization: str = "length",
        proj: bool = False,
        average_windows: bool = True,
        average_epochs: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """Calculates the power spectral density of the data with multitapers,
        using the implementation in MNE 'time_frequency.psd_multitaper'.

        The multitaper method involves calculating the spectral density for
        orthogonal tapers and then averaging them together for each
        channel/epoch.

        PARAMETERS
        ----------
        fmin : int | float; default 0
        -   The minimum frequency of interest.

        fmax : int | float; default infinite
        -   The maximum frequency of interest.

        tmin : float | None; default None
        -   The minimum time of interest.

        tmax : float | None; default None
        -   The maximum time of interest.

        bandwidth : float | None; default None
        -   The bandwidth of the multitaper windowing function, in Hz. If
            'None', this is set to a window half-bandwidth of 4.

        adaptive : bool; default False
        -   Whether or not to use adaptive weights to combine the tapered
            spectra into the power spectral density.

        low_bias : bool; default True.
        -   Whether or not to use only tapers with more than 90% spectral
            concentration within bandwidth.

        normalization : str; default "length"
        -   The normalisation strategy to use. If "length", the power spectra is
            normalised by the length of the signal. If "full", the power spectra
            is normalised by the sampling rate and the signal length.

        proj : bool; default False
        -   Whether or not to apply SSP projection vectors.

        average_windows : bool; default True
        -   Whether or not to average power results across data windows.

        average_epochs : bool; default True
        -   Whether or not to average power results across epochs.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel. If '-1', this is set to the
            number of CPU cores. Requires the 'joblib' package.

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
            print("Performing multitaper power analysis on the data.\n")

        self._get_results_multitaper(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            bandwidth=bandwidth,
            adaptive=adaptive,
            low_bias=low_bias,
            normalization=normalization,
            proj=proj,
            average_windows=average_windows,
            average_epochs=average_epochs,
            n_jobs=n_jobs,
        )

        self._processed = True
        self._power_method = "multitaper"
        self.processing_steps["power_multitaper"] = {
            "fmin": fmin,
            "fmax": fmax,
            "tmin": tmin,
            "tmax": tmax,
            "bandwidth": bandwidth,
            "adaptive": adaptive,
            "low_bias": low_bias,
            "normalization": normalization,
            "proj": proj,
            "average_windows": average_windows,
            "average_epochs": average_epochs,
        }

    def _get_results_multitaper(
        self,
        fmin: Union[int, float],
        fmax: float,
        tmin: Union[float, None],
        tmax: Union[float, None],
        bandwidth: Union[float, None],
        adaptive: bool,
        low_bias: bool,
        normalization: str,
        proj: bool,
        average_windows: bool,
        average_epochs: bool,
        n_jobs: int,
    ) -> None:
        """Calculates the power spectral density of the data with multitapers,
        using the implementation in MNE 'time_frequency.psd_multitaper'.

        The multitaper method involves calculating the spectral density for
        orthogonal tapers and then averaging them together for each
        channel/epoch.

        PARAMETERS
        ----------
        fmin : int | float; default 0
        -   The minimum frequency of interest.

        fmax : int | float; default infinite
        -   The maximum frequency of interest.

        tmin : float | None; default None
        -   The minimum time of interest.

        tmax : float | None; default None
        -   The maximum time of interest.

        bandwidth : float | None; default None
        -   The bandwidth of the multitaper windowing function, in Hz. If
            'None', this is set to a window half-bandwidth of 4.

        adaptive : bool; default False
        -   Whether or not to use adaptive weights to combine the tapered
            spectra into the power spectral density.

        low_bias : bool; default True.
        -   Whether or not to use only tapers with more than 90% spectral
            concentration within bandwidth.

        normalization : str; default "length"
        -   The normalisation strategy to use. If "length", the power spectra is
            normalised by the length of the signal. If "full", the power spectra
            is normalised by the sampling rate and the signal length.

        proj : bool; default False
        -   Whether or not to apply SSP projection vectors.

        average_windows : bool; default True
        -   Whether or not to average power results across data windows.

        average_epochs : bool; default True
        -   Whether or not to average power results across epochs.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel. If '-1', this is set to the
            number of CPU cores. Requires the 'joblib' package.
        """
        results = []
        ch_names = self.signal.data[0].ch_names
        ch_types = self.signal.data[0].get_channel_types(picks=ch_names)
        for i, data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"\n---=== Computing power for window {i+1} of "
                    f"{len(self.signal.data)} ===---\n"
                )
            psds, freqs = time_frequency.psd_multitaper(
                inst=data,
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                bandwidth=bandwidth,
                adaptive=adaptive,
                low_bias=low_bias,
                normalization=normalization,
                picks=ch_names,
                proj=proj,
                n_jobs=n_jobs,
                reject_by_annotation=False,
                verbose=self._verbose,
            )
            results.append(
                FillableObject(
                    attrs={
                        "data": psds,
                        "freqs": freqs,
                        "ch_names": ch_names,
                        "ch_types": ch_types,
                    }
                )
            )
        self.results = results

        self._results_dims = ["windows", "epochs", "channels", "frequencies"]
        if average_windows:
            self._average_results_dim("windows")
        if average_epochs:
            self._average_results_dim("epochs")

    def process_morlet(
        self,
        freqs: list[Union[int, float]],
        n_cycles: Union[int, float, list[Union[int, float]]],
        use_fft: bool = False,
        zero_mean: bool = True,
        average_windows: bool = True,
        average_epochs: bool = True,
        average_timepoints: bool = True,
        decim: Union[int, slice] = 1,
        n_jobs: int = 1,
    ) -> None:
        """Calculates a time-frequency representation with Morlet wavelets,
        using the implementation in MNE 'time_frequency.TFR_morlet'.

        PARAMETERS
        ----------
        freqs : list[int | float]
        -   The frequencies, in Hz, to analyse.

        n_cycles : int | float | list[int | float]
        -   The number of cycles to use. If an int or float, this number of
            cycles is used for all frequencies. If a list, each entry should
            correspond to the number of cycles for an individual frequency.

        use_fft : bool; default False
        -   Whether or not to use FFT-based convolution.

        zero_mean : bool; default True
        -   Whether or not to set the mean of the wavelets to 0.

        average_windows : bool; default True
        -   Whether or not to average the results across windows.

        average_epochs : bool; default True
        -   Whether or not to average the results across epochs.

        average_timepoints : bool; default True
        -   Whether or not to average the results across timepoints.

        decim : int | slice : default 1
        -   The decimation factor to use after time-frequency decomposition to
            reduce memory usage. If an int, returns [..., ::decim]. If a slice,
            returns [..., decim].

        n_jobs : int; default 1
        -   The number of jobs to run in paraller. If '-1', it is set to the
            number of CPU cores. Requires the 'joblib' package.

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

        self._get_results_morlet(
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=use_fft,
            zero_mean=zero_mean,
            average_windows=average_windows,
            average_epochs=average_epochs,
            average_timepoints=average_timepoints,
            decim=decim,
            n_jobs=n_jobs,
        )

        self._processed = True
        self._power_method = "morlet"
        self.processing_steps["power_morlet"] = {
            "freqs": freqs,
            "n_cycles": n_cycles,
            "use_fft": use_fft,
            "zero_mean": zero_mean,
            "average_windows": average_windows,
            "average_epochs": average_epochs,
            "average_timepoints": average_timepoints,
            "decim": decim,
        }

    def _get_results_morlet(
        self,
        freqs: list[Union[int, float]],
        n_cycles: Union[int, float, list[Union[int, float]]],
        use_fft: bool,
        zero_mean: bool,
        average_windows: bool,
        average_epochs: bool,
        average_timepoints: bool,
        decim: Union[int, slice],
        n_jobs: int,
    ) -> None:
        """Calculates a time-frequency representation with Morlet wavelets,
        using the implementation in MNE 'time_frequency.TFR_morlet'.

        PARAMETERS
        ----------
        freqs : list[int | float]
        -   The frequencies, in Hz, to analyse.

        n_cycles : int | float | list[int | float]
        -   The number of cycles to use. If an int or float, this number of
            cycles is used for all frequencies. If a list, each entry should
            correspond to the number of cycles for an individual frequency.

        use_fft : bool; default False
        -   Whether or not to use FFT-based convolution.

        zero_mean : bool; default True
        -   Whether or not to set the mean of the wavelets to 0.

        average_windows : bool; default True
        -   Whether or not to average the results across windows.

        average_epochs : bool; default True
        -   Whether or not to average the results across epochs.

        average_timepoints : bool; default True
        -   Whether or not to average the results across timepoints.

        decim : int | slice : default 1
        -   The decimation factor to use after time-frequency decomposition to
            reduce memory usage. If an int, returns [..., ::decim]. If a slice,
            returns [..., decim].

        n_jobs : int; default 1
        -   The number of jobs to run in paraller. If '-1', it is set to the
            number of CPU cores. Requires the 'joblib' package.
        """
        results = []
        for i, data in enumerate(self.signal.data):
            if self._verbose:
                print(
                    f"\n---=== Computing power for window {i+1} of "
                    f"{len(self.signal.data)} ===---\n"
                )
            output = time_frequency.tfr_morlet(
                inst=data,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=use_fft,
                return_itc=False,
                decim=decim,
                n_jobs=n_jobs,
                picks=np.arange(len(self.signal.data[0].ch_names)),
                zero_mean=zero_mean,
                average=False,
                output="power",
                verbose=self._verbose,
            )
            results.append(
                FillableObject(
                    attrs={
                        "data": output.data,
                        "freqs": output.freqs,
                        "ch_names": output.ch_names,
                        "ch_types": output.get_channel_types(
                            picks=output.ch_names
                        ),
                    }
                )
            )
        self.results = results

        self._results_dims = [
            "windows",
            "epochs",
            "channels",
            "frequencies",
            "timepoints",
        ]
        if average_windows:
            self._average_results_dim("windows")
        if average_epochs:
            self._average_results_dim("epochs")
        if average_timepoints:
            self._average_results_dim("timepoints")

    def _average_results_dim(self, dim: str) -> None:
        """Averages results of the analysis across a results dimension.

        PARAMETERS
        ----------
        dim : str
        -   The dimension of the results to average across. Recognised inputs
            are: "window"; "epochs"; "frequencies"; and "timepoints".

        RAISES
        ------
        NotImplementedError
        -   Raised if the dimension is not supported.
        ProcessingOrderError
        -   Raised if the dimension has already been averaged across.
        ValueError
        -   Raised if the dimension is not present in the results to average
            across.
        """
        recognised_inputs = [
            "windows",
            "epochs",
            "frequencies",
            "timepoints",
            "segments",
        ]
        if dim not in recognised_inputs:
            raise NotImplementedError(
                f"The dimension '{dim}' is not supported for averaging. "
                f"Supported dimensions are {recognised_inputs}."
            )
        if getattr(self, f"_{dim}_averaged"):
            raise ProcessingOrderError(
                f"Trying to average the results across {dim}, but this has "
                "already been done."
            )
        if dim not in self.results_dims:
            raise ValueError(
                f"No {dim} are present in the results to average across."
            )

        if dim == "windows":
            n_events = self._average_windows()
        else:
            dim_i = self._results_dims.index(dim) - 1
            n_events = np.shape(self.results[0].data)[dim_i]
            for i, power in enumerate(self.results):
                self.results[i].data = np.mean(power.data, axis=dim_i)
            self._results_dims.pop(dim_i + 1)

        setattr(self, f"_{dim}_averaged", True)
        if self._verbose:
            print(f"\nAveraging the data over {n_events} {dim}.")

    def _average_windows(self) -> int:
        """Averages the power results across windows.

        RETURNS
        -------
        n_windows : int
        -   The number of windows being averaged across.
        """
        n_windows = len(self.results)
        power = []
        for results in self.results:
            power.append(results.data)
        self.results[0].data = np.asarray(power).mean(axis=0)
        self.results = [self.results[0]]

        return n_windows

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
        NotImplementedError
        -   Raised if the requested normalisation type is not supported.
        -   Raised if the length of the results dimensions are greater than the
            maximum number supported.
        ValueError
        -   Raised if the dimension to normalise across is not present in the
            results dimensions.
        -   Raised if a window of results are to be excluded from the
            normalisation around the line noise, but no line noise frequency is
            given.
        """
        supported_norm_types = ["percentage_total"]
        max_supported_n_dims = 2

        if norm_type not in supported_norm_types:
            raise NotImplementedError(
                "Error when normalising the results of the Morlet power "
                f"analysis:\nThe normalisation type '{norm_type}' is not "
                "supported. The supported normalisation types are: "
                f"{supported_norm_types}."
            )

        if len(self.results_dims) > max_supported_n_dims:
            raise NotImplementedError(
                "Error when normalising the results of the Morlet power "
                "analysis:\nCurrently, normalising the values of results with "
                f"at most {max_supported_n_dims} is supported, but the results "
                f"have {len(self.results_dims)} dimensions."
            )

        if within_dim not in self.results_dims:
            raise ValueError(
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
        self.processing_steps[f"power_{self._power_method}_normalisation"] = {
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
        ask_before_overwrite: Union[bool, None] = None,
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
            f"power-{self._power_method}": results.tolist(),
            f"power-{self._power_method}_dimensions": dimensions,
            "freqs": self.results[0].freqs.tolist(),
            "ch_names": self.results[0].ch_names,
            "ch_types": self.results[0].ch_types,
            "ch_coords": self.signal.get_coordinates(),
            "ch_regions": ordered_list_from_dict(
                self.results[0].ch_names, self.extra_info["ch_regions"]
            ),
            "ch_subregions": ordered_list_from_dict(
                self.results[0].ch_names, self.extra_info["ch_subregions"]
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

        PARAMETERS
        ----------
        dimensions : list[str] | None
        -   The order of the dimensions of the results to return. If 'None', the
            current result dimensions are used.

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

        if self._windows_averaged:
            results = self.results[0].data
        else:
            results = []
            for mne_obj in self.results:
                results.append(mne_obj.data)
            results = np.asarray(results)

        if dimensions is not None and dimensions != self.results_dims:
            results = rearrange_axes(
                obj=results, old_order=self.results_dims, new_order=dimensions
            )

        return deepcopy(results)

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        ask_before_overwrite: Union[bool, None] = None,
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
    signal : PowerStandard
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

    def __init__(self, signal: PowerStandard, verbose: bool = True) -> None:
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
        self.freq_bands = None
        self.average_windows = None
        self.show_fit = None
        self.fooof_results = None
        self.freqs = None

    def _sort_inputs(self) -> None:
        """Checks the inputs to the PowerFOOOF object to ensure that they
        match the requirements for processing and assigns inputs.

        RAISES
        ------
        InputTypeError
        -   Raised if the PowerMorlet object input does not contain data in a
            supported format.
        """
        supported_data_dims = [["windows", "channels", "frequencies"]]
        if self.signal._results_dims not in supported_data_dims:
            raise TypeError(
                "Error when applying FOOOF to the power data:\nThe data in the "
                f"power object is in the form {self.signal.results_dims}, but "
                f"only data in the form {supported_data_dims} is supported."
            )
        super()._sort_inputs()

    def process(
        self,
        freq_range: Union[list[int, float], None] = None,
        peak_width_limits: list[Union[int, float]] = [0.5, 12],
        max_n_peaks: Union[int, float] = float("inf"),
        min_peak_height: Union[int, float] = 0,
        peak_threshold: Union[int, float] = 2,
        aperiodic_modes: Union[dict[Union[str, None]], None] = None,
        freq_bands: Union[dict[list[Union[int, float]]], None] = None,
        average_windows: bool = True,
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

        aperiodic_modes : dict[str | None] | None; default None
        -   The mode for fitting the periodic component, can be "fixed" or
            "knee". Each key should be the name of a channel, and the value the
            fitting mode.
        -   If 'None', the user is shown, for each channel individually, the
            results of the different fits and asked to choose the fits to use
            for each channel.
        -   If some entries are 'None', the user is show the results of the
            different fits for these channels and asked to choose the fits to
            use for each of these channels.

        freq_bands : dict[list[int | float]] | None; default None
        -   The frequency bands to analyse in the data for extracting peaks from
            the periodic spectra. The peak with the maximum power in each band
            is taken.
        -   Each key should be the name of the band, and each value a list
            containing the lower and upper frequencies of the band,
            respectively.
        -   If 'None', the entire frequency range is used, and the peak with the
            maximum power across the periodic spectra taken.

        average_windows : bool; default True
        -   Whether or not to average results across windows.

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
            freq_bands=freq_bands,
            average_windows=average_windows,
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
            "average_windows": self.average_windows,
        }

    def _sort_processing_inputs(
        self,
        freq_range: Union[list[int, float], None],
        peak_width_limits: list[Union[int, float]],
        max_n_peaks: Union[int, float],
        min_peak_height: Union[int, float],
        peak_threshold: Union[int, float],
        aperiodic_modes: Union[dict[Union[str, None]], None],
        freq_bands: Union[dict[list[Union[int, float]]], None],
        average_windows: bool,
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

        aperiodic_modes : dict[str | None] | None
        -   The mode for fitting the periodic component, can be "fixed" or
            "knee". Each key should be the name of a channel, and the value the
            fitting mode.
        -   If 'None', the user is shown, for each channel individually, the
            results of the different fits and asked to choose the fits to use
            for each channel.
        -   If some entries are 'None', the user is show the results of the
            different fits for these channels and asked to choose the fits to
            use for each of these channels.

        freq_bands : dict[list[int | float]] | None
        -   The frequency bands to analyse in the data for extracting peaks from
            the periodic spectra. The peak with the maximum power in each band
            is taken.
        -   Each key should be the name of the band, and each value a list
            containing the lower and upper frequencies of the band,
            respectively.
        -   If 'None', the entire frequency range is used, and the peak with the
            maximum power across the periodic spectra taken.

        average_windows : bool
        -   Whether or not to average the results across windows.

        show_fit : bool
        -   Whether or not to show the user the FOOOF model fits for the
            channels.

        RAISES
        ------
        ValueError
        -   Raised if 'aperiodic_modes' are specified and are not provided for
            each window and each channel.
        """

        if freq_range is None:
            freq_range = [
                self.signal.results[0].freqs[0],
                self.signal.results[0].freqs[-1],
            ]
        self.freq_range = freq_range
        self.freqs = self.signal.results[0].freqs

        if aperiodic_modes is None:
            aperiodic_modes = []
            for win_i in range(len(self.signal.results)):
                aperiodic_modes.append({})
                for ch_name in self.signal.results[0].ch_names:
                    aperiodic_modes[win_i][ch_name] = None
        else:
            if len(aperiodic_modes) != len(self.signal.results):
                raise ValueError(
                    "A set of aperiodic modes must be provided for each of the "
                    f"{len(self.signal.results)} data windows."
                )
            for win_i in range(len(self.signal.results)):
                for ch_name in self.signal.results[0].ch_names:
                    if ch_name not in aperiodic_modes[win_i].keys():
                        raise ValueError(
                            f"The channel '{ch_name}' is not present in the "
                            f"aperiodic modes for window {win_i}."
                        )
        self.aperiodic_modes = aperiodic_modes

        if freq_bands is None:
            freq_bands = {"whole_spectrum": freq_range}
        self.freq_bands = freq_bands

        self.peak_width_limits = peak_width_limits
        self.max_n_peaks = max_n_peaks
        self.min_peak_height = min_peak_height
        self.peak_threshold = peak_threshold
        self.average_windows = average_windows
        self.show_fit = show_fit

    def _get_results(
        self,
    ) -> dict[NDArray]:
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
        }
        for win_i, aperiodic_modes in enumerate(self.aperiodic_modes):
            for key in results.keys():
                results[key].append([])
            for ch_name, mode in aperiodic_modes.items():
                ch_i = self.signal.results[win_i].ch_names.index(ch_name)
                data = self.signal.results[win_i].data[ch_i]
                if mode is None:
                    mode = self._choose_aperiodic_mode(
                        data=data, channel_name=ch_name
                    )
                    self.aperiodic_modes[win_i][ch_name] = mode
                result = self._apply_fooof(
                    data=data, channel_name=ch_name, aperiodic_mode=mode
                )
                for key, value in result.items():
                    results[key][win_i].append(value)

        results = self._combine_results_over_channels(results=results)
        if self.average_windows:
            results = self._average_windows(results=results)
        self._windows_averaged = True

        return results

    def _choose_aperiodic_mode(self, data: NDArray, channel_name: str) -> str:
        """Fits the FOOOF model to the data for both the 'fixed' and 'knee'
        aperiodic modes, and asks the user to choose the desired mode.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   The data of the channel for which the aperiodic mode should be
            checked.

        channel_name: str
        -   The name of the channel in the data which should be examined.

        RETURNS
        -------
        str
        -   The aperiodic mode chosen by the user. Will be 'fixed' or 'knee'.
        """
        if self._verbose:
            print(
                "No aperiodic mode has been specified for channel "
                f"'{channel_name}'. Please choose an aperiodic mode to use "
                "after closing the figure."
            )

        fm_fixed = self._fit_fooof_model(data=data, aperiodic_mode="fixed")
        fm_knee = self._fit_fooof_model(data=data, aperiodic_mode="knee")

        self._plot_aperiodic_modes(
            fm_fixed=fm_fixed, fm_knee=fm_knee, channel_name=channel_name
        )

        return self._input_aperiodic_mode_choice()

    def _fit_fooof_model(self, data: NDArray, aperiodic_mode: str) -> FOOOF:
        """Fits a FOOOF model to the power data of a single channel according to
        the pre-specified settings.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   The data to fit the FOOOF model to.

        aperiodic_mode : str
        -   The aperiodic mode to use when fitting the model. Recognised modes
            are 'fixed' and 'knee'.

        RETURNS
        -------
        fooof_model : fooof FOOOF
        -   The fitted FOOOF model.
        """
        fooof_model = FOOOF(
            peak_width_limits=self.peak_width_limits,
            max_n_peaks=self.max_n_peaks,
            min_peak_height=self.min_peak_height,
            peak_threshold=self.peak_threshold,
            aperiodic_mode=aperiodic_mode,
            verbose=self._verbose,
        )
        fooof_model.fit(
            freqs=self.signal.results[0].freqs,
            power_spectrum=data,
            freq_range=self.freq_range,
        )

        return fooof_model

    def _plot_aperiodic_modes(
        self, fm_fixed: FOOOF, fm_knee: FOOOF, channel_name: str
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

        channel_name : str
        -   The name of the channel whose data is being plotted.
        """
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(channel_name)

        fm_fixed.plot(plt_log=False, ax=axs[0, 0])
        fm_fixed.plot(plt_log=True, ax=axs[1, 0])

        fm_knee.plot(plt_log=False, ax=axs[0, 1])
        fm_knee.plot(plt_log=True, ax=axs[1, 1])

        fit_metrics = ["r_squared_", "error_"]
        fixed_mode_title = "Fixed aperiodic mode | "
        knee_mode_title = "Knee aperiodic mode | "
        for metric in fit_metrics:
            fixed_mode_title += (
                f"{metric[:-1]}={round(getattr(fm_fixed, metric), 2)} | "
            )
            knee_mode_title += (
                f"{metric[:-1]}={round(getattr(fm_knee, metric), 2)} | "
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
        -   The chosen aperiodic mode. Can be 'fixed' or 'knee'.
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

    def _apply_fooof(
        self, data: NDArray, channel_name: str, aperiodic_mode: str
    ) -> dict:
        """Fits a FOOOF model to a single channel of data based on the
        pre-specified settings and organises the results.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   The data to apply the FOOOF model to.

        channel_name : str
        -   The name of the channel whose data is being analysed.

        aperiodic_mode : str
        -   The aperiodic mode to use to fit the FOOOF model. Can be 'fixed' or
            'knee'.

        RETURNS
        -------
        dict
        -   The results of the FOOOF analysis.
        """
        if self._verbose:
            print(
                f"Fitting the channel '{channel_name}' with aperiodic mode "
                f"{aperiodic_mode}.\n"
            )
        fooof_model = self._fit_fooof_model(
            data=data, aperiodic_mode=aperiodic_mode
        )
        if self.show_fit:
            self._inspect_model(model=fooof_model, channel_name=channel_name)

        return self._extract_results_from_model(model=fooof_model)

    def _inspect_model(self, model: FOOOF, channel_name: str) -> None:
        """Plots the fitted FOOOF model alongside the goodness-of-fit metrics.

        PARAMETERS
        ----------
        model : FOOOF
        -   The fitted FOOOF model to inspect.

        channel_name : str
        -   The name of the channel whose data is included in the model.
        """
        fig, axs = plt.subplots(2, 1)
        model_title = (
            f"{channel_name} | {model.aperiodic_mode} aperiodic mode | "
        )
        model.plot(plt_log=False, ax=axs[0])
        model.plot(plt_log=True, ax=axs[1])
        fit_metrics = ["r_squared_", "error_"]
        for metric in fit_metrics:
            model_title += (
                f"{metric[:-1]}={round(getattr(model, metric), 2)} | "
            )
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
        aperiodic_params = self._sort_aperiodic_params(
            model.get_results().aperiodic_params
        )
        peak_params = self._get_fband_peaks(model.get_results().peak_params)
        return {
            "periodic_component": model._peak_fit,
            "aperiodic_component": model._spectrum_peak_rm,
            "r_squared": model.get_results().r_squared,
            "error": model.get_results().error,
            "aperiodic_params": aperiodic_params,
            "peak_params": peak_params,
        }

    def _sort_aperiodic_params(self, aperiodic_params: NDArray) -> NDArray:
        """Adds an additional entry to the aperiodic parameters of a FOOOF model
        if the 'fixed' aperiodic fitting mode was used.

        If the 'fixed' mode is used, the aperiodic parameters consist of
        [offset, exponent], however is the 'knee' mode is used, the parameters
        are [offset, knee, exponent]. To ensure that the parameters for both
        fitting modes have the same length, an extra entry with the value 'NaN'
        corresponding to the knee entry is added to parameters for the 'fixed'
        mode.

        PARAMETERS
        ----------
        aperiodic_params : numpy ndarray
        -   Aperiodic parameters extracted from a FOOOF model.

        RETURNS
        -------
        sorted_params : numpy ndarray
        -   The aperiodic parameters, modified if appropriate.

        RAISES
        ------
        ValueError
        -   Raised if 'aperiodic_params' does not have length 2 or 3.
        """
        if len(aperiodic_params) == 2:
            sorted_params = [aperiodic_params[0], np.nan, aperiodic_params[1]]
        elif len(aperiodic_params) == 3:
            sorted_params = aperiodic_params.copy()
        else:
            raise ValueError(
                "Aperiodic parameters extracted from a FOOOF model can only "
                "have lengths 2 or 3, but the input has length "
                f"{len(aperiodic_params)}."
            )

        return sorted_params

    def _get_fband_peaks(self, peak_params: NDArray) -> NDArray:
        """Extracts the parameters of the power peaks from the aperiodic spectra
        for individual frequency bands.

        PARAMETERS
        ----------
        peak_params : numpy ndarray
        -   The peak parameters extracted from the FOOOF model.

        RETURNS
        -------
        numpy ndarray
        -   The parameters of the peaks ([central frequency, power, bandwidth])
            for each frequency band.
        -   If no peak is present for a given band, all parameters are 'NaN'.
        """
        fband_peak_params = []
        for band in self.freq_bands.values():
            fband_peak_params.append(
                get_band_peak(
                    peak_params=peak_params,
                    band=band,
                    select_highest=True,
                    threshold=None,
                    thresh_param="PW",
                )
            )

        return np.asarray(fband_peak_params)

    def _combine_results_over_channels(
        self, results: dict[list]
    ) -> dict[NDArray]:
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
        for key, value in results.items():
            results[key] = np.asarray(value)

        return results

    def _average_windows(self, results: dict[NDArray]) -> dict[NDArray]:
        """Averages the results across windows.

        PARAMETERS
        ----------
        results : dict[numpy ndarray]
        -   The results of the FOOOF analysis extracted from the models.

        RETURNS
        -------
        averaged_results : dict[numpy array]
        -   The results averaged across windows.
        """
        averaged_results = deepcopy(results)
        for key, value in results.items():
            averaged_results[key] = [np.nanmean(value, axis=0)]

        return averaged_results

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Union[bool, None] = None,
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

    def get_results(self, return_lists: bool = False) -> dict:
        """Returns the results dictionary.

        PARAMETERS
        ----------
        return_lists : bool; default False
        -   Whether or not to return the results as lists, or numpy ndarrays.

        RETURNS
        -------
        dict
        -   The results dictionary of the FOOOF analysis.
        """
        results = deepcopy(self.results)
        for key, value in results.items():
            if return_lists:
                value = [val.tolist() for val in value]
            if self._windows_averaged:
                value = value[0]
            results[key] = value

        return results

    def results_as_dict(self) -> dict:
        """Organises the results and additional information into a dictionary.

        RETURNS
        dict
        -   A dictionary of results and additional information.
        """
        ch_names = self.signal.results[0].ch_names
        results = self.get_results(return_lists=True)
        return {
            "power-fooof_periodic": results["periodic_component"],
            "power-fooof_aperiodic": results["aperiodic_component"],
            "power-fooof_band_names": list(self.freq_bands.keys()),
            "power-fooof_band_peaks": results["peak_params"],
            "r_squared": results["r_squared"],
            "error": results["error"],
            "freqs": self.freqs.tolist(),
            "ch_names": ch_names,
            "ch_types": self.signal.signal.data[0].get_channel_types(
                picks=ch_names
            ),
            "ch_coords": self.signal.signal.get_coordinates(picks=ch_names),
            "ch_regions": ordered_list_from_dict(
                ch_names, self.extra_info["ch_regions"]
            ),
            "ch_subregions": ordered_list_from_dict(
                ch_names, self.extra_info["ch_subregions"]
            ),
            "ch_hemispheres": ordered_list_from_dict(
                ch_names, self.extra_info["ch_hemispheres"]
            ),
            "ch_reref_types": ordered_list_from_dict(
                ch_names, self.extra_info["ch_reref_types"]
            ),
            "samp_freq": self.signal.signal.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.signal.data[0].info["subject_info"],
        }

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        ask_before_overwrite: Union[bool, None] = None,
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

    def save_aperiodic_modes(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        ask_before_overwrite: Union[bool, None] = None,
    ) -> None:
        """Saves the aperiodic modes used in the FOOOF analysis as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the modes should be saved.

        ftype : str | None; default None
        -   The filetype of the file that will be saved, without the leading
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
            to_save={"aperiodic_modes": self.aperiodic_modes},
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )
