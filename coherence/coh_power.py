"""Classes for performing power analysis on data.

CLASSES
-------
PowerMorlet
-   Performs power analysis on preprocessed data using Morlet wavelets.
"""


from copy import deepcopy
from typing import Any, Optional, Union

import csv
import json
import pickle

from mne import time_frequency

import numpy as np
import pandas as pd

from coh_dtypes import realnum
from coh_exceptions import (
    ChannelOrderError,
    DuplicateEntryError,
    InputTypeError,
    MissingEntryError,
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_handle_entries import (
    check_master_entries_in_sublists,
    check_sublist_entries_in_master,
    check_duplicates_list,
    check_matching_entries,
    ordered_list_from_dict,
)
from coh_handle_files import check_ftype_present
from coh_processing_methods import ProcMethod
from coh_saving import check_before_overwrite
import coh_signal


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
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:

        # Initialises inputs of the Analysis object.
        self.signal = deepcopy(signal)
        self._verbose = verbose
        self._sort_inputs()

        # Initialises aspects of the Analysis object that will be filled with
        # information as the data is processed.
        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.extra_info = deepcopy(self.signal.extra_info)
        self.power = None
        self.itc = None
        self.power_dims = None
        self._power_dims_sorted = None
        self.itc_dims = None
        self._itc_dims_sorted = None

        # Initialises aspects of the Analysis object that indicate which methods
        # have been called (starting as 'False'), which can later be updated.
        self._processed = False
        self._itc_returned = False
        self._epochs_averaged = False
        self._power_timepoints_averaged = False
        self._itc_timepoints_averaged = False
        self._power_in_dataframe = False
        self._itc_in_dataframe = False

    def _sort_inputs(self) -> None:
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing.

        RAISES
        ------
        InputTypeError
        -   Raised if the Signal object input does not contain epoched data,
            which is necessary for power and connectivity analyses.
        """

        if self.signal._getattr("_epoched") is False:
            raise InputTypeError(
                "The provided Signal object does not contain epoched data. "
                "Epoched data is required for power and connectivity analyses."
            )

    def _update_processing_steps(self, step_value: dict) -> None:
        """Updates the 'power_morlet' entry of the 'processing_steps'
        dictionary of the PowerMorlet object with new information.

        PARAMETERS
        ----------
        step_value : dict
        -   A dictionary where the keys are the analysis setting names and the
            values the settings used.
        """

        self.processing_steps["power_morlet"] = step_value

    def _check_identical_ch_orders(self) -> None:
        """Checks to make sure that the order of the channels (and thus, the
        data) in the preprocessed data, power data, and (optionally) inter-
        trial coherence data is identical.

        RAISES
        ------
        ChannelOrderError
        -   Raised if the order of the names of the channels does not match.
        """

        if not check_matching_entries(
            objects=[self.signal.data.ch_names, self.power.ch_names]
        ):
            raise ChannelOrderError(
                "The order of channel names in the preprocessed data and in "
                "the Morlet power data do not match.\nThis should only have "
                "occurred if you re-ordered the channels of these datasets "
                "separately of one another."
            )

        if self._itc_returned:
            if not check_matching_entries(
                objects=[self.signal.data.ch_names, self.itc.ch_names]
            ):
                raise ChannelOrderError(
                    "The order of channel names in the preprocessed data and "
                    "in the inter-trial coherence data do not match.\nThis "
                    "should only have occurred if you re-ordered the channels "
                    "of these datasets separately of one another."
                )

    def _check_vars_present(
        self, master_list: list[str], sublists: list[list[str]]
    ) -> None:
        """Checks to make sure the variables in the variable order list are all
        present in the identical and unique variable lists and that the
        identical and unique variable lists are specified in the variable
        order list.

        PARAMETERS
        ----------
        master_list : list[Any]
        -   A master list of values. Here the variable order list.

        sublists : list[list[Any]]
        -   A list of sublists of values. Here the identical and unique
            variable lists.
        """

        all_present, absent_entries = check_master_entries_in_sublists(
            master_list=master_list, sublists=sublists, allow_duplicates=False
        )
        if not all_present:
            raise MissingEntryError(
                "Error when trying to convert the results of the Morlet power "
                "analysis into a DataFrame:\nThe following columns "
                f"{absent_entries} do not have any data."
            )

        all_present, absent_entries = check_sublist_entries_in_master(
            master_list=master_list, sublists=sublists, allow_duplicates=False
        )
        if not all_present:
            raise MissingEntryError(
                "Error when trying to convert the results of the Morlet power "
                "analysis into a DataFrame:\nThe following columns "
                f"{absent_entries} have not been accounted for when ordering "
                "the columns of the DataFrame."
            )

    def _set_df_identical_vars(
        self, var_names: list[str], var_values: dict[Any]
    ) -> dict:
        """Sets the variables which have identical values regardless of the
        channel from which the data is coming.

        PARAMETERS
        ----------
        var_names : list[str]
        -   Names of the variables with identical values.

        var_values : dict[Any]
        -   Dictionary where the keys are the variable names and the values are
            the values of the variables which are identical across channels.

        RETURNS
        -------
        dict[Any]
        -   Dictionary of key:value pairs where the keys are the variable names
            and the values a list of identical entries for the corresponding
            key.
        """

        n_entries = len(self.signal.data.ch_names)

        return {name: [var_values[name]] * n_entries for name in var_names}

    def _set_df_unique_vars(self, var_names: list[str]) -> dict:
        """Sets the variables which have unique values depending on the
        channel from which the data is coming.

        PARAMETERS
        ----------
        var_names : list[str]
        -   The names of the variables whose values should be collected.

        RETURNS
        -------
        unique_vars : dict
        -   Dictionary of key:value pairs where the keys are the variable names
            and the values a list of entries for the corresponding key ordered
            based on the channels in the processed data.
        """

        unique_vars = {}
        ch_names = self.power.ch_names

        for name in var_names:
            if name == "ch_name":
                unique_vars[name] = ch_names
            elif name == "reref_type":
                unique_vars[name] = [
                    self.extra_info["rereferencing_types"][ch_name]
                    for ch_name in ch_names
                ]
            elif name == "ch_region":
                unique_vars[name] = [
                    self.extra_info["ch_regions"][ch_name]
                    for ch_name in ch_names
                ]
            elif name == "ch_coords":
                unique_vars[name] = self.signal.get_coordinates()
            elif name == "freqs":
                unique_vars[name] = [list(self.power.freqs)] * len(ch_names)
            elif name == "power":
                unique_vars[name] = list(self.power.data)
            elif name == "itc":
                unique_vars[name] = list(self.itc.data)
            else:
                raise UnavailableProcessingError(
                    "Error when converting the Morlet power data to a "
                    f"DataFrame:\nThe variable '{name}' is not recognised."
                )

        return unique_vars

    def _combine_df_vars(
        self, identical_vars: dict[Any], unique_vars: dict[Any]
    ) -> dict:
        """Combines identical and unique variables together into a single
        dictionary.

        PARAMETERS
        ----------
        identical_vars : dict[Any]
        -   Dictionary in which the keys are the names of the variables whose
            values are identical across channels, and the values the variables'
            corresponding values.

        unique_vars : dict[Any]
        -   Dictionary in which the keys are the names of the variables whose
            values are different across channels, and the values the variables'
            corresponding values.

        RETURNS
        -------
        combined_vars : dict[Any]
        -   Dictionary containing the identical and unique variables.

        RAISES
        ------
        DuplicateEntryError
        -   Raised if a variable is listed multiple times within the identical
            and/or unique variables.
        """

        combined_vars = identical_vars | unique_vars

        duplicates, duplicate_values = check_duplicates_list(
            values=combined_vars.keys()
        )
        if duplicates:
            raise DuplicateEntryError(
                "Error when converting the Morlet power analysis results into "
                f"a DataFrame:\nThe DataFrame column(s) {duplicate_values} are "
                "repeated."
            )

        return combined_vars

    def results_to_dataframe(self) -> None:
        """Converts the results of the processing (power and if applicable,
        inter-trial coherence) into DataFrames.
        """

        self._power_to_dataframe()
        if self._itc_returned:
            self._itc_to_dataframe()

    def _to_dataframe(
        self,
        var_order: list[str],
        identical_var_names: list[str],
        unique_var_names: list[str],
    ) -> pd.DataFrame:
        """Converts the processed data into a pandas DataFrame.

        PARAMETERS
        ----------
        var_order : list[str]
        -   The order the variables should take in the DataFrame.

        identical_var_names : dict[Any]
        -   The names of the variables whose values do not depend on the
            channel.

        unique_var_names : dict[Any]
        -   The names of the variables whose values depend on the channel.
        """

        identical_vars = self._set_df_identical_vars(
            identical_var_names, self.extra_info["metadata"]
        )
        unique_vars = self._set_df_unique_vars(unique_var_names)
        combined_vars = self._combine_df_vars(identical_vars, unique_vars)

        return pd.DataFrame.from_dict(combined_vars, orient="columns").reindex(
            columns=var_order
        )

    def _power_to_dataframe(self) -> None:
        """Converts the results of the Morlet wavelet power analysis into a
        pandas DataFrame.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the results are already in a DataFrame.

        ChannelOrderError
        -   Raised if the order of the channel names in the preprocessed data
            and analysis results do not match.
        """

        if self._power_in_dataframe:
            raise ProcessingOrderError(
                "Error when converting the power results of the Morlet wavelet "
                "analysis to a DataFrame:\nThese results have already been "
                "converted to a DataFrame."
            )

        if not check_matching_entries(
            objects=[self.signal.data.ch_names, self.power.ch_names]
        ):
            raise ChannelOrderError(
                "The order of channel names in the preprocessed data and in "
                "the Morlet power data do not match.\nThis should only have "
                "occurred if you re-ordered the channels of these datasets "
                "separately of one another."
            )

        if self._verbose:
            print("Converting the power results into a DataFrame.")

        var_order = [
            "cohort",
            "sub",
            "med",
            "stim",
            "task",
            "ses",
            "run",
            "ch_name",
            "reref_type",
            "ch_coords",
            "ch_region",
            "freqs",
            "power",
        ]
        identical_var_names = [
            "cohort",
            "sub",
            "med",
            "stim",
            "task",
            "ses",
            "run",
        ]
        unique_var_names = [
            "ch_name",
            "reref_type",
            "ch_coords",
            "ch_region",
            "freqs",
            "power",
        ]
        self._check_vars_present(
            var_order, [identical_var_names, unique_var_names]
        )

        self.power = self._to_dataframe(
            var_order, identical_var_names, unique_var_names
        )

        self._power_in_dataframe = True

    def _itc_to_dataframe(self) -> None:
        """Converts the results of the inter-trial coherence analysis into a
        pandas DataFrame.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the results are already in a DataFrame.

        ChannelOrderError
        -   Raised if the order of the channel names in the preprocessed data
            and analysis results do not match.
        """

        if self._itc_in_dataframe:
            raise ProcessingOrderError(
                "Error when converting the inter-trial coherence results of "
                "the Morlet wavelet analysis to a DataFrame:\nThese results "
                "have already been converted to a DataFrame."
            )

        if not check_matching_entries(
            objects=[self.signal.data.ch_names, self.itc.ch_names]
        ):
            raise ChannelOrderError(
                "The order of channel names in the preprocessed data and in "
                "the inter-trial coherence data do not match.\nThis should "
                "only have occurred if you re-ordered the channels of these "
                "datasets separately of one another."
            )

        if self._verbose():
            print("Converting the power results into a DataFrame.")

        var_order = [
            "cohort",
            "sub",
            "med",
            "stim",
            "task",
            "ses",
            "run",
            "ch_name",
            "reref_type",
            "ch_coords",
            "ch_region",
            "power",
        ]
        identical_var_names = [
            "cohort",
            "sub",
            "med",
            "stim",
            "task",
            "ses",
            "run",
        ]
        unique_var_names = [
            "ch_name",
            "reref_type",
            "ch_coords",
            "ch_region",
            "freqs",
            "itc",
        ]
        self._check_vars_present(
            var_order, [identical_var_names, unique_var_names]
        )

        self.itc = self._to_dataframe(
            var_order, identical_var_names, unique_var_names
        )

        self._itc_in_dataframe = True

    def _assign_result(
        self,
        result: Union[
            time_frequency.EpochsTFR, tuple[time_frequency.AverageTFR]
        ],
        itc_returned: bool,
    ) -> None:
        """Assigns the result(s) of the Morlet power analysis.

        PARAMETERS
        ----------
        result :
        -   The result of mne.time_frequency.tfr_morlet.

        itc_returned : bool
        -   States whether inter-trial coherence was also returned from the
            Morlet power analysis.
        """

        if itc_returned:
            self.power = result[0]
            self.itc = result[1]
            self._itc_returned = True
        else:
            self.power = result
            self._itc_returned = False

    def _average_timepoints_power(self) -> None:
        """Averages power results of the analysis across timepoints.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the power data has already been averaged across
            timepoints.
        """

        if self._power_timepoints_averaged:
            raise ProcessingOrderError(
                "Trying to average the data across timepoints, but this has "
                "already been done."
            )

        n_timepoints = np.shape(self.power.data)[-1]
        self.power.data = np.mean(self.power.data, axis=-1)

        self._power_timepoints_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_timepoints} timepoints.")

    def _average_timepoints_itc(self) -> None:
        """Averages inter-trial coherence results of the analysis across
        timepoints.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if coherence data was not returned from the analysis.

        ProcessingOrderError
        -   Raised if the coherence data has already been averaged across
            timepoints.
        """

        if not self._itc_returned:
            raise UnavailableProcessingError(
                "Trying to average the inter-trial coherence data across "
                "timepoints, but this data was not returned from the analysis."
            )
        if self._itc_timepoints_averaged:
            raise ProcessingOrderError(
                "Trying to average the inter-trial coherence data across "
                "timepoints, but this has already been done."
            )

        n_timepoints = np.shape(self.itc.data)[-1]
        self.itc.data = np.mean(self.itc.data, axis=-1)

        self._itc_timepoints_averaged = True
        if self._verbose:
            print(
                f"Averaging the inter-trial coherence data over {n_timepoints} "
                "timepoints."
            )

    def _establish_power_dims(self) -> None:
        """Establishes the dimensions of the power data and notes how they
        should be rearranged before saving."""

        if self._epochs_averaged:
            if self._power_timepoints_averaged:
                self.power_dims = ["channels", "frequencies"]
                self._power_dims_sorted = self.power_dims
            else:
                self.power_dims = ["channels", "frequencies", "timepoints"]
                self._power_dims_sorted = [
                    "channels",
                    "timepoints",
                    "frequencies",
                ]
        else:
            if self._power_timepoints_averaged:
                self.power_dims = ["epochs", "channels", "frequencies"]
                self._power_dims_sorted = ["channels", "epochs", "frequencies"]
            else:
                self.power_dims = [
                    "epochs",
                    "channels",
                    "frequencies",
                    "timepoints",
                ]
                self._power_dims_sorted = [
                    "channels",
                    "epochs",
                    "timepoints",
                    "frequencies",
                ]

    def _establish_itc_dims(self) -> None:
        """Establishes the dimensions of the inter-trial coherence data and
        notes how they should be rearranged before saving.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if inter-trial coherence data was not returned from the
            analysis.
        """

        if not self._itc_returned:
            raise UnavailableProcessingError(
                "Trying to re-order the dimensions of the inter-trial "
                "coherence data, but this data was not returned from the "
                "analysis."
            )

        if self._itc_timepoints_averaged:
            self.itc_dims = ["channels", "frequencies"]
            self._itc_dims_sorted = self.itc_dims
        else:
            self.itc_dims = ["channels", "frequencies", "timepoints"]
            self._itc_dims_sorted = ["channels", "timepoints", "frequencies"]

    def _get_result(
        self,
        freqs: list[realnum],
        n_cycles: Union[int, list[int]],
        use_fft: bool = False,
        return_itc: bool = True,
        decim: Union[int, slice] = 1,
        n_jobs: int = 1,
        picks: Union[list[int], None] = None,
        zero_mean: bool = True,
        average_epochs: bool = True,
        output: str = "power",
    ) -> Union[
        time_frequency.EpochsTFR,
        tuple[time_frequency.AverageTFR],
        tuple[time_frequency.EpochsTFR, time_frequency.AverageTFR],
    ]:
        """Gets the result of the Morlet power analysis.
        -   Allows the user to calculate inter-trial coherence without having
            power data averaged across epochs, overcoming a limitation in the
            mne.time_frequency.tfr_morlet implementation in which if return_itc
            is True, average (over epochs) must also be True.

        PARAMETERS
        ----------
        freqs : list[realnum]
        -   The frequencies in Hz to analyse.

        n_cycles : int | list[int]
        -   The number of cycles globally (if int) or for each frequency (if
            list[int]).

        use_fft : bool; default False
        -   Whether or not to perform the fft based convolution.

        return_itc : bool; default True
        -   Whether or not to retirn inter-trial coherence in addition to power.
            Must be false for evoked data.

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

        average_epochs : bool; default True
        -   If True, averages the power across epochs. If False, returns
            separate power values for each epoch.

        output : str; default 'power'
        -   Can be 'power' or 'complex'. If 'complex', average must be False.

        RETURNS
        -------
        result : time_frequency.EpochsTFR | tuple[time_frequency.AverageTFR] |
        tuple[time_frequency.EpochsTFR, time_frequency.AverageTFR]
        -   The result of the Morlet power analysis (optionally alongside
            inter-trial coherence).
        -   If return_itc if False and average_epochs is False, the result is
            the power as an MNE time_frequency.EpochsTFR object with the
            dimensions [epochs x channels x frequencies x timepoints].
        -   If return_itc if False and average_epochs is True, the result is
            the power as an MNE time_frequency.EpochsTFR object with the
            dimensions [channels x frequencies x timepoints].
        -   If return_itc is True and average_epochs if False, the result is a
            tuple with the power as an MNE time_frequency.EpochsTFR object with
            the dimensions [epochs x channels x frequencies x timepoints] and
            the inter-trial coherence as an MNE time_frequency.AverageTFR object
            with the dimensions [channels x frequencies x timepoints].
        -   If return_itc is True and average_epochs if True, the result is a
            tuple with the power as an MNE time_frequency.AverageTFR object with
            the dimensions [channels x frequencies x timepoints] and the
            inter-trial coherence as an MNE time_frequency.AverageTFR object
            with the dimensions [channels x frequencies x timepoints].
        """

        if return_itc:
            if average_epochs:
                result = time_frequency.tfr_morlet(
                    inst=self.signal.data,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=use_fft,
                    return_itc=return_itc,
                    decim=decim,
                    n_jobs=n_jobs,
                    picks=picks,
                    zero_mean=zero_mean,
                    average=average_epochs,
                    output=output,
                    verbose=self._verbose,
                )
            else:
                power = time_frequency.tfr_morlet(
                    inst=self.signal.data,
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
                _, itc = time_frequency.tfr_morlet(
                    inst=self.signal.data,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=use_fft,
                    return_itc=return_itc,
                    decim=decim,
                    n_jobs=n_jobs,
                    picks=picks,
                    zero_mean=zero_mean,
                    average=True,
                    output=output,
                    verbose=self._verbose,
                )
                result = (power, itc)
        else:
            result = time_frequency.tfr_morlet(
                inst=self.signal.data,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=use_fft,
                return_itc=return_itc,
                decim=decim,
                n_jobs=n_jobs,
                picks=picks,
                zero_mean=zero_mean,
                average=average_epochs,
                output=output,
                verbose=self._verbose,
            )

        return result

    def process(
        self,
        freqs: list[realnum],
        n_cycles: Union[int, list[int]],
        use_fft: bool = False,
        return_itc: bool = True,
        decim: Union[int, slice] = 1,
        n_jobs: int = 1,
        picks: Union[list[int], None] = None,
        zero_mean: bool = True,
        average_epochs: bool = True,
        average_timepoints_power: bool = True,
        average_timepoints_itc: bool = False,
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

        return_itc : bool; default True
        -   Whether or not to retirn inter-trial coherence in addition to power.
            Must be false for evoked data.

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

        average_epochs : bool; default True
        -   If True, averages the power across epochs. If False, returns
            separate power values for each epoch.

        average_timepoints_power : bool; default True
        -   If True, averages the power across timepoints within each epoch. If
            False, returns separate power values for each timepoint in the
            epoch.

        average_timepoints_itc : bool; default False
        -   If True, averages the inter-trial coherence across timepoints. If
            False, returns separate coherence values for each timepoint. If this
            is set to True, return_itc must also be True.

        output : str; default 'power'
        -   Can be 'power' or 'complex'. If 'complex', average must be False.

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
            if return_itc:
                print(
                    "Performing Morlet wavelet power analysis on the data and "
                    "returning the inter-trial coherence."
                )
            else:
                print("Performing Morlet wavelet power analysis on the data.")

        result = self._get_result(
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=use_fft,
            return_itc=return_itc,
            decim=decim,
            n_jobs=n_jobs,
            picks=picks,
            zero_mean=zero_mean,
            average_epochs=average_epochs,
            output=output,
        )

        self._epochs_averaged = average_epochs

        self._assign_result(result, return_itc)
        if average_timepoints_power:
            self._average_timepoints_power()
        if average_timepoints_itc:
            self._average_timepoints_itc()

        self._establish_power_dims()
        if return_itc:
            self._establish_itc_dims()
            self._itc_returned = True

        self._processed = True
        self._update_processing_steps(
            {
                "freqs": freqs,
                "n_cycles": n_cycles,
                "use_fft": use_fft,
                "return_itc": return_itc,
                "decim": decim,
                "n_jobs": n_jobs,
                "picks": picks,
                "zero_mean": zero_mean,
                "average_epochs": average_epochs,
                "average_timepoints_power": average_timepoints_power,
                "average_timepoints_itc": average_timepoints_itc,
                "output": output,
            }
        )

    def _prepare_results_for_saving(  # SUPER
        self,
        results: np.array,
        results_structure: Optional[list[str]],
        rearrange: Optional[list[str]],
    ) -> list:
        """Extracts the power and inter-trial coherence results from
        mne.time_frequency.EpochsTFR or .AverageTFR objects as a list in
        preparation for saving.

        PARAMETERS
        ----------
        results : numpy array
        -   The results of the power or inter-trial coherence analysis.

        results_structure : list[str] | None; default None
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
                [results_structure.index(axis) for axis in rearrange],
            )

        return extracted_results.tolist()

    def _save_as_json(self, to_save: dict, fpath: str) -> None:  # SUPER
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

    def _save_as_csv(self, to_save: dict, fpath: str) -> None:  # SUPER
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

    def _save_as_pkl(self, to_save: Any, fpath: str) -> None:  # SUPER
        """Pickles and saves information in any format.

        PARAMETERS
        ----------
        to_save : Any
        -   Information that will be saved.

        fpath : str
        -   Location where the data should be saved.
        """

        with open(fpath, "wb") as file:
            pickle.dump(to_save, file)

    def save_object(  # SUPER
        self, fpath: str, ask_before_overwrite: Optional[bool] = None
    ) -> None:
        """Saves the PowerMorlet object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """

        if not check_ftype_present(fpath):
            fpath += ".pkl"

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose
        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            self._save_as_pkl(self, fpath)

    def save_results(  # SUPER
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the results (power and inter-trial coherence, if applicable)
        and additional information as a file.

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
        UnavailableProcessingError
        -   Raised if the given format for saving the file is in an unsupported
            format.
        """

        power = self._prepare_results_for_saving(
            self.power.data,
            results_structure=self.power_dims,
            rearrange=self._power_dims_sorted,
        )

        objects = [self.signal.data.ch_names, self.power.ch_names]
        if self._itc_returned:
            objects.append(self.itc.ch_names)
        if not check_matching_entries(objects=objects):
            raise ChannelOrderError(
                "Error when trying to save the results of the Morlet wavelet "
                "power analysis:\nThe channel names in the preprocessed data "
                "and in the results do not match."
            )

        to_save = {
            "power": power,
            "power_structure": self._power_dims_sorted,
            "ch_names": self.signal.data.ch_names,
            "ch_types": self.signal.data.get_channel_types(),
            "ch_coords": self.signal.get_coordinates(),
            "ch_regions": ordered_list_from_dict(
                self.power.ch_names, self.extra_info["ch_regions"]
            ),
            "reref_types": ordered_list_from_dict(
                self.power.ch_names, self.extra_info["reref_types"]
            ),
            "samp_freq": self.signal.data.info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data.info["subject_info"],
        }
        if self._itc_returned:
            itc = self._prepare_results_for_saving(
                self.itc.data,
                results_structure=self.itc_dims,
                rearrange=self._itc_dims_sorted,
            )
            to_save["itc"] = itc
            to_save["itc_structure"] = self._itc_dims_sorted

        super.save_results(
            to_save=to_save,
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
        )

        if self._verbose:
            print(f"Saving the raw signals to:\n'{fpath}'.")

    def save(
        self,
        fpath: str,
        ask_before_overwrite: Union[bool, None] = None,
    ) -> None:
        """Saves the processing results to a specified location.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath where the results will be saved.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists. If False, the user is not asked to
            confirm this and it is done automatically. By default, this is set
            to None, in which case the value of the verbosity when the
            PowerMorlet object was instantiated is used.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        if self._verbose:
            print(f"Saving the morlet power results to:\n'{fpath}'.")

        attr_to_save = ["power", "processing_steps"]
        if self._itc_returned:
            attr_to_save.append("itc")

        super().save(fpath, self, attr_to_save, ask_before_overwrite)


# class PowerFOOOF(ProcMethod):
