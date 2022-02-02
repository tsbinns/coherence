from abc import ABC, abstractmethod
import mne
import numpy as np

from coh_check_entries import CheckLengthsList
from coh_dtypes import realnum
from coh_exceptions import EntryLengthError




class Reref(ABC):
    """Abstract class for rereferencing data in mne.io.Raw objects.

    METHODS
    -------
    rereference (abstract)
    -   Rereferences the data in an mne.io.Raw object.

    SUBCLASSES
    ----------
    RerefBipolar
    -   Bipolar rereferences data in an mne.io.Raw object.

    RerefCAR
    -   Common-average rereferences data in an mne.io.Raw object.

    RerefPseudo
    -   Psuedo rereferences data in an mne.io.Raw object.
    -   This allows you to alter characteristics of the mne.io.Raw object (e.g.
        channel coordinates) and assign a rereferencing type to the channels
        without altering the data.
    -   This is useful if e.g. the channels were already hardware rereferenced.
    """

    @abstractmethod
    def _check_input_lengths(self,
        lengths_to_check : list[list]
        ) -> int:
        """Checks that the lengths of the entries (representing the features of
        channels that will be rereferenced, e.g. channel names, coordinates,
        etc...) within a list are of the same length.
        -   This length corresponds to the number of channels in the
            rereferenced data.

        PARAMETERS
        ----------
        lengths_to_check : list[list]
        -   List containing the entries whose lengths should be checked.

        RETURNS
        -------
        int
        -   The length of the list's entries, corresponding to the number of
            channels in the rereferenced data.
        -   Only returned if the lengths of the entries in the list are equal,
            else an error is raised before.

        RAISES
        ------
        EntryLengthError
        -   Raised if the lengths of the list's entries are nonidentical.
        """

        equal_lengths, n_channels = CheckLengthsList(
            lengths_to_check, [None]
        ).identical()

        if not equal_lengths:
            raise EntryLengthError(
                "Error when reading rereferencing settings:\nThe length of "
                "entries within the settings dictionary are not identical:\n"
                f"{n_channels}"
            )

        return n_channels


    @abstractmethod
    def _sort_raw(self,
        raw: mne.io.Raw,
        chs_to_analyse : list[str]
        ) -> mne.io.Raw:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the subclasses' method.

        PARAMETERS
        ----------
        raw : mne.io.Raw
        -   The mne.io.Raw object to drop channels from.

        chs_to_analyse : list[str]
        -   List containing the names of the channels in mne.io.Raw to retain.

        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object with only the rereferencing-relevant channels
            remaining.
        """

        raw.drop_channels(
            [name for name in raw.info['ch_names']
            if name not in chs_to_analyse]
        )

        return raw.reorder_channels(chs_to_analyse)


    @abstractmethod
    def _sort_ch_names_new(self) -> None:
        pass


    @abstractmethod
    def _sort_ch_types_new(self) -> None:
        pass


    @abstractmethod
    def _sort_reref_types(self) -> None:
        pass


    @abstractmethod
    def _sort_ch_coords_new(self) -> None:
        pass


    @abstractmethod
    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.
        -   Implemented in the subclasses' method.
        """

    @abstractmethod
    def _data_from_raw(self,
        raw: mne.io.Raw
        ) -> tuple[
            np.ndarray, mne.Info, list[str], list[list[realnum]]
        ]:
        """Extracts components of an mne.io.Raw object and returns them.

        PARAMETERS
        ----------
        raw : mne.io.Raw
        -   The mne.io.Raw object whose data and information should be
            extracted.

        RETURNS
        -------
        np.array
        -   Array of the data with shape [n_channels, n_timepoints].

        mne.Info
        -   Information taken from the mne.io.Raw object.

        list[str]
        -   List of channel names taken from the mne.io.Raw object corresponding
            to the channels in the data array.

        list[list[int or float]]
        -   List of channel coordinates taken from the mne.io.Raw object, with
            each channel's coordinates given in a sublist containing the x, y,
            and z coordinates.
        """

        return (raw.get_data(reject_by_annotation='omit').copy(),
                raw.info.copy(),
                raw.info['ch_names'].copy(),
                raw._get_channel_positions().copy().tolist())


    @abstractmethod
    def _raw_from_data(self,
        data: np.ndarray,
        data_info: mne.Info,
        ch_coords: list[list[realnum]]
        ) -> mne.io.Raw:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.

        PARAMETERS
        ----------
        data : np.array
        -   Array of the rereferenced data with shape [n_channels x
            n_timepoints].

        data_info : mne.Info
        -   Information about the data in 'data'.

        ch_coords : empty list or list[list[int or float]] | optional, default
        []
        -   Coordinates of the channels, with each channel's coordinates
            contained in a sublist consisting of the x, y, and z coordinates.

        RETURNS
        -------
        raw : mne.io.Raw
        -   The constructed mne.io.Raw object containing the rereferenced data.
        """

        raw = mne.io.RawArray(data, data_info)
        if ch_coords != []:
            raw._set_channel_positions(ch_coords, data_info['ch_names'])

        return raw


    @abstractmethod
    def _store_rereference_types(self,
        ch_names: list[str],
        reref_types: list[str]
        ) -> dict[str, str]:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that will become the dictionary keys.

        reref_types : list[str]
        -   The types of the rereferencing applied, corresponding to the
            channels in 'ch_names', that will become the dictionary values.

        RETURNS
        -------
        dict[str, str]
        -   Dictionary of key:value pairs consisting of channel name :
            rereferencing type.
        """

        return {ch_names[i]: reref_types[i] for i in range(len(ch_names))}


    @abstractmethod
    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced.
        -   Implemented in the subclasses' method.
        """


    @abstractmethod
    def _set_data(self) -> None:
        """Rereferences the data.
        -   Implemented in the subclasses' method.
        """


    @abstractmethod
    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   Implemented in the subclasses.
        """


    @abstractmethod
    def _set_data_info(self,
        ch_names: list[str],
        ch_types: list[str],
        old_info: mne.Info
        ) -> mne.Info:
        """Creates an mne.Info object containing information about the newly
        rereferenced data.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels in the rereferenced data.

        ch_types : list[str]
        -   The types of channels in the rereferenced data, according to those
            recognised by MNE, corresponding to the channels in 'ch_names'.

        old_info : mne.Info
        -   The mne.Info object from the unrereferenced mne.io.Raw object to
            extract still-relevant information from to set in the new mne.Info
            object.

        RETURNS
        -------
        new_info : mne.Info
        -   mne.Info object containing information about the newly rereferenced
            data.
        """

        new_info = mne.create_info(ch_names, old_info['sfreq'], ch_types)
        do_not_overwrite = ['ch_names', 'chs', 'nchan']
        for key, value in old_info.items():
            if key not in do_not_overwrite:
                new_info[key] = value

        return new_info


    @abstractmethod
    def rereference(self) -> None:
        """Rereferences the data in an mne.io.Raw object.
        -   Implemented in the subclasses' method.
        """
