"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""

import csv
import json
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from numpy.typing import NDArray
from typing import Union
import numpy as np
import coh_signal
from coh_exceptions import (
    EntryLengthError,
    MissingFileExtensionError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_files import check_ftype_present, identify_ftype
from coh_saving import check_before_overwrite


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

    def _save_as_json(self, to_save: dict, fpath: str) -> None:
        """Saves entries in a dictionary as a json file.

        PARAMETERS
        ----------
        to_save : dict
        -   Dictionary in which the keys represent the names of the entries in
            the json file, and the values represent the corresponding values.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "w", encoding="utf8") as file:
            json.dump(to_save, file)

    def _save_as_csv(self, to_save: dict, fpath: str) -> None:
        """Saves entries in a dictionary as a csv file.

        PARAMETERS
        ----------
        to_save : dict
        -   Dictionary in which the keys represent the names of the entries in
            the csv file, and the values represent the corresponding values.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "wb") as file:
            save_file = csv.writer(file)
            save_file.writerow(to_save.keys())
            save_file.writerow(to_save.values())

    def _save_as_pkl(
        self, to_save: Union["ProcMethod", dict], fpath: str
    ) -> None:
        """Pickles and saves information in any format.

        PARAMETERS
        ----------
        to_save : ProcMethod | dict
        -   Information that will be saved.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "wb") as file:
            pickle.dump(to_save, file)

    def _save_object(
        self,
        to_save: "ProcMethod",
        fpath: str,
        ask_before_overwrite: bool,
        verbose: bool,
    ) -> None:
        """Saves the ProcMethod subclass object as a .pkl file.

        PARAMETERS
        ----------
        to_save : ProcMethod
        -   ProcMethod subclass object to save.

        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.

        verbose : bool
        -   Whether or not to print a note of the saving process.
        """

        if not check_ftype_present(fpath):
            fpath += ".pkl"

        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            self._save_as_pkl(to_save=to_save, fpath=fpath)

            if verbose:
                print(f"Saving the analysis object to:\n{fpath}")

    def _save_results(
        self,
        to_save: dict,
        fpath: str,
        ftype: Union[str, None],
        ask_before_overwrite: bool,
        verbose: bool,
    ) -> None:
        """Saves the results (power and inter-trial coherence, if applicable)
        and additional information as a file.

        PARAMETERS
        ----------
        to_save : dict
        -   The results to save.

        fpath : str
        -   Location where the data should be saved.

        ftype : str
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.

        ask_before_overwrite : bool
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.

        verbose : bool
        -   Whether or not to print a note of the saving process.

        RAISES
        ------
        UnidenticalEntryError
        -   Raised if the filetype in the filepath and the specified filetype do
            not match.

        MissingFileExtensionError
        -   Raised if no filetype is present in the filetype and one is not
            specified.

        UnavailableProcessingError
        -   Raised if the given format for saving the file is in an unsupported
            format.
        """

        if check_ftype_present(fpath) and ftype is not None:
            fpath_ftype = identify_ftype(fpath)
            if fpath_ftype != ftype:
                raise UnidenticalEntryError(
                    "Error when trying to save the results of the analysis:\n "
                    f"The filetypes in the filepath ({fpath_ftype}) and in the "
                    f"requested filetype ({ftype}) do not match."
                )
        elif check_ftype_present(fpath) and ftype is None:
            ftype = identify_ftype(fpath)
        elif not check_ftype_present(fpath) and ftype is not None:
            fpath += ftype
        else:
            raise MissingFileExtensionError(
                "Error when trying to save the results of the analysis:\nNo "
                "filetype is given in the filepath and no filetype has been "
                "specified."
            )

        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            if ftype == "json":
                self._save_as_json(to_save, fpath)
            elif ftype == "csv":
                self._save_as_csv(to_save, fpath)
            elif ftype == "pkl":
                self._save_as_pkl(to_save, fpath)
            else:
                raise UnavailableProcessingError(
                    "Error when trying to save the analysis results:\nThe "
                    f"{ftype} format for saving is not supported."
                )

            if verbose:
                print(f"Saving the analysis results to:\n'{fpath}'.")


def get_freq_band_results_of_channels(
    freqs: list[Union[int, float]],
    freq_bands: list[list[Union[int, float]]],
    results: Union[list[list[Union[int, float]]], NDArray],
) -> tuple[
    list[list[Union[int, float]]],
    list[list[Union[int, float]]],
    list[list[Union[int, float]]],
]:
    """Calculates the results averaged across specified frequency bands for
    multiple channels of data.

    PARAMETERS
    ----------
    freqs : list[int | float]
    -   The frequencies, in Hz, in the results.

    freq_bands : list[list[int | float]]
    -   The frequency bands to process.
    -   Each entry is a list containing the lower and upper bounds of the
        frequency bands to analyse, in Hz.

    results : list[list[int | float]] | numpy array
    -   The results to process, consisting of a list of lists, where each
        list corresponds to the results of a single channel, of which each
        element corresponds to the frequencies in 'freqs'.

    RETURNS
    -------
    freq_bands_avg : list[int | float]
    -   The average values in each frequency band.

    freq_bands_max : list[int | float]
    -   The maximum values in each frequency band.

    freq_bands_max_freq : list[int | float]
    -   The frequencies at which the maximum values in each frequency band
        occur.
    """

    freq_bands_avg = []
    freq_bands_max = []
    freq_bands_max_freq = []

    for ch_results in results:
        avg_vals, max_vals, max_freq_vals = get_freq_band_results_of_channel(
            freqs=freqs, freq_bands=freq_bands, results=ch_results
        )
        freq_bands_avg.append(avg_vals)
        freq_bands_max.append(max_vals)
        freq_bands_max_freq.append(max_freq_vals)

    return freq_bands_avg, freq_bands_max, freq_bands_max_freq


def get_freq_band_results_of_channel(
    freqs: list[Union[int, float]],
    freq_bands: list[list[Union[int, float]]],
    results: Union[list[Union[int, float]], NDArray],
) -> tuple[
    list[Union[int, float]],
    list[Union[int, float]],
    list[Union[int, float]],
]:
    """Calculates the results averaged across specified frequency bands for
    a single channel of data.

    PARAMETERS
    ----------
    freqs : list[int | float]
    -   The frequencies, in Hz, in the results.

    results : list[int | float] | numpy array
    -   The results to process. Can be one- or multi-dimensional.
    -   If multi-dimensional, the 0th axis is assumed to correspond to the
        frequencies in 'freqs'.

    freq_bands : list[list[int | float]]
    -   The frequency bands to process.
    -   Each entry is a list containing the lower and upper bounds of the
        frequency bands to analyse, in Hz.

    RETURNS
    -------
    freq_bands_avg : list[int | float]
    -   The average values in each frequency band.

    freq_bands_max : list[int | float]
    -   The maximum values in each frequency band.

    freq_bands_max_freq : list[int | float]
    -   The frequencies at which the maximum values in each frequency band
        occur.

    RAISES
    ------
    EntryLengthError
    -   Raised if the length of the 0th axis of 'results' is not equal to the
        number of frequencies in 'freqs'.
    """

    if np.shape(results)[0] != len(freqs):
        raise EntryLengthError(
            "Error when trying to calculate the results within frequency "
            "bands:\nThe 0th axis of the results (length: "
            f"{np.shape(results[0])}) must correspond to the frequencies "
            f"(length: {len(freqs)})."
        )

    freq_bands_avg = []
    freq_bands_max = []
    freq_bands_max_freq = []

    for band in freq_bands:
        try:
            band_idcs = [freqs.index(freq) for freq in band]
        except:
            raise ValueError(
                "Error when trying to calculate the frequency band results:\n"
                f"The frequency band {band_idcs} is not present in the results "
                f"with frequency range [{freqs[0]}, {freqs[-1]}]."
            )

        band_values = results[band_idcs[0] : band_idcs[1] + 1].tolist()
        freq_bands_avg.append(float(np.mean(band_values)))
        freq_bands_max.append(float(np.max(band_values)))
        freq_bands_max_freq.append(
            freqs[band_values.index(freq_bands_max[-1]) + band_idcs[0]]
        )

    return freq_bands_avg, freq_bands_max, freq_bands_max_freq
