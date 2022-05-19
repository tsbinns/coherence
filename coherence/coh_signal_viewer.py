"""Class for viewing raw signals, and adding annotations to these signals.

CLASSES
-------
SignalViewer
-   Allows the user to view non-epoched signals and any pre-existing
    annotations, as well as add new annotations.
"""

import coh_signal
import mne
from matplotlib import pyplot as plt
from coh_exceptions import UnsupportedFileExtensionError
from coh_handle_files import (
    check_annotations_empty,
    check_ftype_present,
    identify_ftype,
)
from coh_saving import check_before_overwrite


class SignalViewer:
    """Allows the user to view non-epoched signals and any pre-existing
    annotations, as well as add new annotations.
    -   Supports the addition of a special annotation "END", which is converted
        to a "BAD" annotation spanning from the start of the "END" annotation
        until the end of the recording.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The raw or preprocessed (but not epoched) signals to add annotations to.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    plot
    -   Plots the raw signals, and if loaded, any pre-existing annotations.

    load_annotations
    -   Loads annotations from a csv file.

    save_annotations
    -   Saves the annotations as a csv file.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:

        # Initialises inputs of the object.
        self.signal = signal
        self._verbose = verbose
        self._sort_inputs()

    def _sort_inputs(self) -> None:
        """Checks that the inputs to the object match the requirements for
        processing

        RAISES
        ------
        InputTypeError
        -   Raised if the data contained in the Signal object has been windowed
            or epoched.
        """

        if self.signal._windowed:
            raise TypeError(
                "Error when trying to instantiate the Annotations object:\n"
                "The data in the Signal object being used has been windowed. "
                "Only non-windowed data is supported."
            )
        if self.signal._epoched:
            raise TypeError(
                "Error when trying to instantiate the Annotations object:\n"
                "The data in the Signal object being used has been epoched. "
                "Only non-epoched data is supported."
            )

    def _sort_fpath(self, fpath: str) -> str:
        """Checks whether the provided filepath for loading or saving
        annotations.
        -   If a filetype is present, checks if it is a supported type (i.e.
            '.csv').
        -   If a filetype is not present, add a '.csv' filetype ending.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath to check

        RETURNS
        -------
        fpath : str
        -   The checked filepath, with filetype added if necessary.

        RAISES
        ------
        UnsupportedFileExtensionError
        -   Raised if the 'fpath' contains a file extension that is not '.csv'.
        """

        if check_ftype_present(fpath):
            fpath_ftype = identify_ftype(fpath)
            supported_ftypes = ["csv"]
            if fpath_ftype != "csv":
                raise UnsupportedFileExtensionError(
                    "Error when trying to save the annotations:\nThe filetype "
                    f"{fpath_ftype} is not supported. Annotations can only be "
                    "saved as filetypes: "
                    f"{[ftype for ftype in supported_ftypes]}"
                )
        else:
            fpath += ".csv"

        return fpath

    def load_annotations(self, fpath: str) -> None:
        """Loads pre-existing annotations for the signals from a csv file.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath to load the annotations from.
        """

        fpath = self._sort_fpath(fpath=fpath)

        if check_annotations_empty(fpath):
            print("There are no events to read from the annotations file.")
        else:
            self.signal.data[0].set_annotations(mne.read_annotations(fpath))

        if self._verbose:
            print(
                f"Loading {len(self.signal.data[0].annotations)} annotations "
                f"from the filepath:\n'{fpath}'"
            )

    def _sort_annotations(self) -> None:
        """Checks the annotations and converts any named 'END' into a 'BAD'
        annotation that spans from the start of the 'END' annotation to the end
        of the recording."""

        end_time = self.signal.data[0].times[-1]
        for i, label in enumerate(self.signal.data[0].annotations.description):
            if label == "END":
                self.signal.data[0].annotations.duration[i] = (
                    end_time - self.signal.data[0].annotations.onset[i]
                )
                self.signal.data[0].annotations.description[i] = "BAD_"

    def plot(self) -> None:
        """Plots the raw signals along with the loaded annotations, if
        applicable."""

        self.signal.data[0].plot(scalings="auto", show=False)
        plt.tight_layout()
        plt.show(block=True)

    def save_annotations(
        self, fpath: str, ask_before_overwrite: bool = True
    ) -> None:
        """Saves the annotations to a csv file.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath to save the annotations to.

        ask_before_overwrite : bool; default True
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        """

        self._sort_annotations()

        fpath = self._sort_fpath(fpath=fpath)

        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            self.signal.data[0].annotations.save(fname=fpath, overwrite=True)

            if self._verbose:
                print(
                    f"Saving {len(self.signal.data[0].annotations)} annotation(s) "
                    f"to:\n'{fpath}'"
                )
