"""A class for processing a Signal object.

CLASSES
-------
Analysis
-   Class for processing a Signal object for power and connectivity analyses.
"""




from copy import deepcopy
from typing import Any, Union
from mne import time_frequency

from coh_dtypes import realnum
from coh_exceptions import (
    InputTypeError, MissingAttributeError, ProcessingOrderError
    )
from coh_signal import Signal




class Analysis:
    """ Class for processing a Signal object for power and connectivity
    analyses.

    PARAMETERS
    ----------
    data : Signal object
    -   A Signal object containing the pre-processed data for further
        processing.

    verbose : bool | Optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    """

    def __init__(self,
        signal : Signal,
        verbose : bool = True,
        ) -> None:

        # Initialises inputs of the Analysis object.
        self.signal = deepcopy(signal)
        self._verbose = verbose
        self._sort_inputs()

        # Initialises aspects of the Analysis object that will be filled with
        # information as the data is processed.
        self.processing_steps = []
        self.data = None

        # Initialises aspects of the Analysis object that indicate which methods
        # have been called (starting as 'False'), which can later be updated.
        self._power = False
        self._FOOOF = False
        self._coh = False
        self._icoh = False


    def _updateattr(self,
        attribute: str,
        value: Any
        ) -> None:
        """Updates aspects of the Signal object that indicate which methods
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


    def _getattr(self,
        obj: Any,
        attribute: str
        ) -> Any:
        """Gets aspects of the input object that indicate which methods have
        been called.
        -   The aspects must have already been instantiated.

        PARAMETERS
        ----------
        obj : Any
        -   The object whose aspect should be examined.

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

        if not hasattr(obj, attribute):
            raise MissingAttributeError(
                f"Error when attempting to get an attribute of {obj}:\nThe "
                f"attribute {attribute} does not exist, and so its value "
                "cannot be updated."
            )

        return getattr(obj, attribute)


    def _sort_inputs(self) -> None:
        """Checks the inputs to the Analysis object to ensure that they match
        the requirements for processing.

        RAISES
        ------
        InputTypeError
        -   Raised if the Signal object input does not contain epoched data,
            which is necessary for power and connectivity analyses.
        """

        if self._getattr(self.signal, '_epoched') is False:
            raise InputTypeError(
                "The provided Signal object does not contain epoched data. "
                "Epoched data is required for power and connectivity analyses."
                )


    def power_morlet(self,
        freqs : list[realnum],
        n_cycles : Union[int, list[int]],
        use_fft : bool = False,
        return_itc : bool = True,
        decim : Union[int, slice] = 1,
        n_jobs : int = 1,
        picks : Union[list[int], None] = None,
        zero_mean : bool = True,
        average : bool = True,
        output : str = 'power',
        verbose : Union[bool, None] = None,
        ) -> None:
        """Performs Morlet wavelet power analysis using the implementation in
        mne.time_frequency.tfr_morlet.

        PARAMETERS
        ----------
        freqs : list[realnum]
        -   The frequencies in Hz to analyse.

        n_cycles : int or list[int]
        -   The number of cycles globally (if int) or for each frequency (if
            list[int]).

        use_fft : bool | default False
        -   Whether or not to perform the fft based convolution.

        return_itc : bool | default True
        -   Whether or not to retirn inter-trial coherence in addition to power.
            Must be false for evoked data.

        decim : int or slice | default 1
        -   Decimates data following time-frequency decomposition. Returns
            data[..., ::decim] if int. Returns data[..., decim]. Warning: may
            create decimation artefacts.

        n_jobs : int | default 1
        -   The number of jobs to run in parallel on the CPU. If -1, it is set
            to the number of CPU cores. Requires the joblib package.

        picks : list[int] or None | default None
        -   The indices of the channels to decompose. If None, all good data
            channels are decomposed.

        zero_mean : bool | default True
        -   Gives the wavelet a mean of 0.

        average : bool | default True
        -   If True, averages the power across epochs. If False, returns
            separate power values for each epoch.

        output : str | default 'power'
        -   Can be 'power' or 'complex'. If 'complex', average must be False.

        verbose : bool or None | default None
        -   Verbosity of the logging output. If None, the default verbosity
            level is used.

        RETURNS
        -------
        power : mne.time_frequency.AverageTFR or mne.time_frequency.EpochsTFR
        -   If average is True, power is averaged across epochs. If average is
            False, power is returned for each epoch.

        itc : mne.time_frequency.AverageTFR or mne.time_frequency.EpochsTFR
        -   The inter-trial coherence. If average is True, coherence is averaged
            across epochs. If average is False, coherence is returned for each
            epoch. Only returned if return_itc is True.
        """

        if self._getattr(self, '_morlet_power'):
            print(
                "Morlet wavelet analysis has already been performed on the "
                "data. The previous results will be overwritten."
            )

        result = time_frequency.tfr_morlet(
            self.signal, freqs, n_cycles, use_fft=use_fft,
            return_itc=return_itc, decim=decim, n_jobs=n_jobs, picks=picks,
            zero_mean=zero_mean, average=average, output=output,
            verbose=verbose,
        )
        if return_itc is True:
            power = result[0]
            itc = result[1]
        else:
            power = result