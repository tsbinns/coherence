"""Classes for performing power analysis on data.

CLASSES
-------
PowerMorlet
-   Performs power analysis on preprocessed data using Morlet wavelets.
"""




from copy import deepcopy
from typing import Any, Union
from mne import time_frequency
import numpy as np

from coh_dtypes import realnum
from coh_exceptions import (
    InputTypeError, MissingAttributeError, ProcessingOrderError,
    UnavailableProcessingError
)
from coh_processing_methods import ProcMethod
import coh_signal




class PowerMorlet(ProcMethod):
    """Performs power analysis on preprocessed data using Morlet wavelets.

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

    def __init__(self,
        signal: coh_signal.Signal,
        verbose: bool = True
        ) -> None:

        # Initialises inputs of the Analysis object.
        self.signal = deepcopy(signal)
        self._verbose = verbose
        self._sort_inputs()

        # Initialises aspects of the Analysis object that will be filled with
        # information as the data is processed.
        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.power = None
        self.itc = None
        self.power_dims = None
        self.itc_dims = None

        # Initialises aspects of the Analysis object that indicate which methods
        # have been called (starting as 'False'), which can later be updated.
        self._processed = False
        self._itc_returned = False
        self._epochs_averaged = False
        self._power_timepoints_averaged = False
        self._itc_timepoints_averaged = False
        self._power_dims_sorted = False
        self._itc_dims_sorted = False


    def _getattr(self,
        attribute: str
        ) -> Any:
        """Gets aspects of the input object that indicate which methods have
        been called.
        -   The aspects must have already been instantiated.

        PARAMETERS
        ----------
        attribute : str
        -   The name of the aspect whose value should be returned.

        RETURNS
        -------
        Any
        -   The value of the aspect.

        RAISES
        ------
        MissingAttributeError
        -   Raised if the user attempts to access an attribute that has not been
            instantiated.
        """

        if not hasattr(self, attribute):
            raise MissingAttributeError(
                f"Error when attempting to get an attribute of {self}:\nThe "
                f"attribute {attribute} does not exist, and so its value "
                "cannot be updated."
            )

        return getattr(self, attribute)


    def _updateattr(self,
        attribute: str,
        value: Any
        ) -> None:
        """Updates aspects of the object that indicate which methods
        have been called.
        -   The aspects must have already been instantiated.

        PARAMETERS
        ----------
        attribute : str
        -   The name of the aspect to update.

        value : Any
        -   The value to update the attribute with.

        RAISES
        ------
        MissingAttributeError
        -   Raised if the user attempts to update an attribute that has not been
            instantiated in '_instantiate_attributes'.
        """

        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise MissingAttributeError(
                f"Error when attempting to update an attribute of {self}:\nThe "
                f"attribute {attribute} does not exist, and so cannot be "
                "updated."
            )


    def _sort_inputs(self) -> None:
        """Checks the inputs to the Analysis object to ensure that they match
        the requirements for processing.

        RAISES
        ------
        InputTypeError
        -   Raised if the Signal object input does not contain epoched data,
            which is necessary for power and connectivity analyses.
        """

        if self.signal._getattr('_epoched') is False:
            raise InputTypeError(
                "The provided Signal object does not contain epoched data. "
                "Epoched data is required for power and connectivity analyses."
            )


    def _update_processing_steps(self,
        step_value: dict
        ) -> None:
        """Updates the 'power_morlet' entry of the 'processing_steps'
        dictionary of the PowerMorlet object with new information.

        PARAMETERS
        ----------
        step_value : dict
        -   A dictionary where the keys are the analysis setting names and the
            values the settings used.
        """

        self.processing_steps['power_morlet'] = step_value


    def _to_dataframe(self) -> None:
        """Converts the processed data into a pandas DataFrame."""

        var_order = [
            'cohort', 'sub', 'med', 'stim', 'task', 'ses', 'run', 'ch_name',
            'ch_coords', 'ch_region', 'power'
        ]
        if self._itc_returned:
            var_order.append('itc')

        




    def _assign_result(self,
        result: Union[
            time_frequency.EpochsTFR, tuple[time_frequency.AverageTFR]
        ],
        itc_returned: bool
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
            print(f'Averaging the data over {n_timepoints} timepoints.')


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


    def _sort_power_dims(self) -> None:
        """Sorts the dimensions of the power data.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the dimensions of the power data have already been
            re-ordered.
        """

        if self._power_dims_sorted:
            raise ProcessingOrderError(
                "Trying to re-order the dimensions of the power data, but this "
                "has already been done."
            )

        if self._epochs_averaged:
            if self._power_timepoints_averaged:
                self.power_dims = ['channels', 'frequencies']
            else:
                self.power.data = np.transpose(self.power.data, (0, 2, 1))
                self.power_dims = ['channels', 'timepoints', 'frequencies']
        else:
            if self._power_timepoints_averaged:
                self.power.data = np.transpose(self.power.data, (1, 0, 2))
                self.power_dims = ['channels', 'epochs', 'frequencies']
            else:
                self.power.data = np.transpose(self.power.data, (1, 0, 3, 2))
                self.power_dims = [
                    'channels', 'epochs', 'timepoints', 'frequencies'
                ]

        self._power_dims_sorted = True


    def _sort_itc_dims(self) -> None:
        """Sorts the dimensions of the inter-trial coherence data.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if coherence data was not returned from the analysis.

        ProcessingOrderError
        -   Raised if the dimensions of the power data have already been
            re-ordered.
        """

        if not self._itc_returned:
            raise UnavailableProcessingError(
                "Trying to re-order the dimensions of the inter-trial "
                "coherence data, but this data was not returned from the "
                "analysis."
            )
        if self._itc_dims_sorted:
            raise ProcessingOrderError(
                "Trying to re-order the dimensions of the inter-trial "
                "coherence data, but this has already been done."
            )

        if self._itc_timepoints_averaged:
            self.itc_dims = ['channels', 'frequencies']
        else:
            self.itc.data = np.transpose(self.itc.data, (0, 2, 1))
            self.itc_dims = ['channels', 'timepoints', 'frequencies']

        self._itc_dims_sorted = True


    def _get_result(self,
        freqs: list[realnum],
        n_cycles: Union[int, list[int]],
        use_fft: bool = False,
        return_itc: bool = True,
        decim: Union[int, slice] = 1,
        n_jobs: int = 1,
        picks: Union[list[int], None] = None,
        zero_mean: bool = True,
        average_epochs: bool = True,
        output: str = 'power'
        ) -> Union[
            time_frequency.EpochsTFR, tuple[time_frequency.AverageTFR],
            tuple[time_frequency.EpochsTFR, time_frequency.AverageTFR]
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
                    verbose=self._verbose
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
                    verbose=self._verbose
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
                    verbose=self._verbose
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
                verbose=self._verbose
            )

        return result


    def process(self,
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
        output: str = 'power'
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

        if self._getattr('_processed'):
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

        self._sort_power_dims()
        if return_itc:
            self._sort_itc_dims()

        self._to_dataframe()

        self._updateattr('_processed', True)
        self._update_processing_steps({
            'freqs': freqs,
            'n_cycles': n_cycles,
            'use_fft': use_fft,
            'return_itc': return_itc,
            'decim': decim,
            'n_jobs': n_jobs,
            'picks': picks,
            'zero_mean': zero_mean,
            'average_epochs': average_epochs,
            'average_timepoints_power': average_timepoints_power,
            'average_timepoints_itc': average_timepoints_itc,
            'output': output,
        })


    def save(self,
        fpath: str,
        ask_before_overwrite: Union[bool, None] = None
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

        attr_to_save = ['power', 'processing_steps']
        if self._itc_returned:
            attr_to_save.append('itc')

        super().save(fpath, self, attr_to_save, ask_before_overwrite)



#class PowerFOOOF(ProcMethod):