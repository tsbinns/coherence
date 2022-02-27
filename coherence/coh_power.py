"""Classes for performing power analysis on data.

CLASSES
-------
PowerMorlet
-   Performs power analysis on preprocessed data using Morlet wavelets.
"""




from copy import deepcopy
from typing import Any, Union
from mne import time_frequency

from coh_dtypes import realnum
from coh_exceptions import (
    InputTypeError, MissingAttributeError, ProcessingOrderError
)
from coh_processing_methods import ProcMethod
import coh_signal
from coherence.coh_exceptions import ProcessingOrderError




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

        # Initialises aspects of the Analysis object that indicate which methods
        # have been called (starting as 'False'), which can later be updated.
        self._processed = False


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

        dependent_vars = {
            'power': []
        }
        if self._itc_returned:
            dependent_vars['itc'] = []
        independent_vars = {
            'med': [],
            'stim': [],
            'task': []
        }
        id_vars = {
            'cohort': [],
            'sub': [],
            'run': [],
            'ch_name': [],
            'ch_coords': [],
            'region': []
        }

        var_order = [
            'cohort', 'sub', 'med', 'stim', 'task', 'run', 'ch_name',
            'ch_coords', 'region', *dependent_vars
        ]


    def process(self,
        freqs: list[realnum],
        n_cycles: Union[int, list[int]],
        use_fft: bool = False,
        return_itc: bool = True,
        decim: Union[int, slice] = 1,
        n_jobs: int = 1,
        picks: Union[list[int], None] = None,
        zero_mean: bool = True,
        average: bool = True,
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

        average : bool; default True
        -   If True, averages the power across epochs. If False, returns
            separate power values for each epoch.

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
            print("Performing Morlet wavelet power analysis on the data.")

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
            average=average,
            output=output,
            verbose=self._verbose
        )
        if return_itc is True:
            self.power = result[0]
            self.itc = result[1]
            self._itc_returned = True
        else:
            self.power = result
            self._itc_returned = False
        self._to_dataframe()

        self._updateattr('_processed', True)
        self._update_processing_steps({
            'freqs': freqs, 'n_cycles': n_cycles, 'use_fft': use_fft,
            'return_itc': return_itc, 'decim': decim, 'n_jobs': n_jobs,
            'picks': picks, 'zero_mean': zero_mean, 'average': average,
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