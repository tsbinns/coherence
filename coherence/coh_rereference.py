"""An abstract class for rereferencing data in an mne.io.Raw object.

CLASSES
-------
Reref : abstract base class
-   Abstract class for rereferencing data in an mne.io.Raw object.
-   This class should not be called directly. Instead, its subclasses should be
    called from their respective files.
"""




from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional
import mne
import numpy as np

from coh_check_entries import CheckLengthsList
from coh_dtypes import realnum
from coh_exceptions import ChannelTypeError, EntryLengthError




class Reref(ABC):
    """Abstract class for rereferencing data in an mne.io.Raw object.
    -   This class should not be called directly. Instead, its subclasses should
        be called from their respective files.

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



class RerefBipolar(Reref):
    """Bipolar rereferences data in an mne.io.Raw object.
    -   Subclass of the abstract class Reref.

    PARAMETERS
    ----------
    raw : mne.io.Raw
    -   The mne.io.Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the mne.io.Raw object to rereference.

    ch_names_new : list[str or None] or None | default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   Missing values (None) will be set based on 'ch_names_old'.

    ch_types_new : list[str or None] or None | default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   Missing values (None) will be set based on the types of channels in
        'ch_names_old'.

    reref_types : list[str or None] or None | default None
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   Missing values (None) will be set as 'CAR', common-average
        rereferencing.

    ch_coords_new : list[list[int or float] or None] or None | default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]', the coordinates of the channels in
        'ch_names_old' in the mne.io.Raw object are used.
    -   If some sublists are '[]', those channels for which coordinates are
        given are used, whilst those channels for which the coordinates are
        missing have their coordinates taken from the mne.io.Raw object
        according to the corresponding channel in 'ch_names_old'.

    METHODS
    -------
    rereference
    -   Rereferences the data in an mne.io.Raw object.
    """

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list[str],
        ch_names_new: Optional[list[Optional[str]]] = None,
        ch_types_new: Optional[list[Optional[str]]] = None,
        reref_types: Optional[list[Optional[str]]] = None,
        ch_coords_new: Optional[list[Optional[list[realnum]]]] = None
        ) -> None:

        self.raw = raw
        self._data_from_raw()
        self._ch_names_old = ch_names_old
        self._index_old_channels()
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._reref_types = reref_types

        self._sort_inputs()


    def _check_input_lengths(self) -> None:
        """Checks that the lengths of the entries (representing the features of
        channels that will be rereferenced, e.g. channel names, coordinates,
        etc...) within a list are of the same length.
        -   This length corresponds to the number of channels in the
            rereferenced data.
        -   Implemented in the parent class' method.
        """

        self._n_channels = super()._check_input_lengths(
            [self._ch_names_old, self._ch_names_new, self._ch_types_new,
            self._reref_types, self._ch_coords_new]
        )


    def _sort_raw(self) -> None:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the parent class' method.
        """

        chs_to_analyse = np.unique(
            [name for names in self._ch_names_old for name in names]
        ).tolist()

        self.raw = super()._sort_raw(self.raw, chs_to_analyse)


    def _check_ch_names_old(self) -> None:
        """Checks that two channel names (i.e. an anode and a cathode) are given
        for each new, rereferenced channels.

        RAISES
        ------
        EntryLengthError
        -   Raised if two channel names (i.e. an anode and a cathode) in the
            original data are not provided for each new, bipolar-rereferenced
            channel being produced.
        """

        if not CheckLengthsList(self._ch_names_old).equals_n(2):
            raise EntryLengthError(
                "Error when bipolar rereferencing data:\nThis must involve "
                "two, and only two channels of data, but the rereferencing "
                "settings specify otherwise."
            )


    def _sort_ch_names_new(self) -> None:
        """Resolves any missing entries for the names of the new, rereferenced
        channels, generating names based on the channels being rereferenced.
        """

        if self._ch_names_new is None:
            for ch_names in self._ch_names_old:
                self._ch_names_new.append('-'.join(name for name in ch_names))
        elif any(item is None for item in self._ch_names_new):
            for i, ch_name in enumerate(self._ch_names_new):
                if ch_name is None:
                    self._ch_names_new[i] = '-'.join(
                        name for name in self._ch_names_old[i]
                    )


    def _sort_reref_types(self) -> None:
        """Resolves any missing entries from the rereference types of the new
        channels, setting them to 'bipolar'.
        """

        if self._reref_types is None:
            self._reref_types = ['bipolar' for i in range(self._n_channels)]
        elif any(item is None for item in self._reref_types):
            for i, reref_type in enumerate(self._reref_types):
                if reref_type is None:
                    self._reref_types[i] = 'bipolar'


    def _sort_ch_types_new(self) -> None:
        """Resolves any missing entries from the channels types of the new
        channels, based on the types of channels they will be rereferenced from.

        RAISES
        ------
        ChannelTypeMismatch
        -   Raised if two channels which are being rereferenced into a new
            channel do not have the same channel type and the type of this new
            channel has not been specified.
        """

        if self._ch_types_new is None:
            self._ch_types_new = []
            for ch_names in self._ch_names_old:
                ch_types_old = list(np.unique(
                    self.raw.get_channel_types(ch_names))
                )
                if len(ch_types_old) == 1:
                    self._ch_types_new.append(ch_types_old[0])
                else:
                    raise ChannelTypeError(
                        "Error when trying to bipolar rereference data:\nNo "
                        "channel types of rereferenced channels have been "
                        "specified, and they cannot be generated based on the "
                        "channels being rereferenced as they are of different "
                        "channel types."
                    )
        elif any(item is None for item in self._ch_types_new):
            for i, ch_type in enumerate(self._ch_types_new):
                if ch_type is None:
                    ch_types_old = list(np.unique(
                        self.raw.get_channel_types(self._ch_names_old[i])
                    ))
                    if len(ch_types_old) == 1:
                        self._ch_types_new.append(ch_types_old[0])
                    else:
                        raise ChannelTypeError(
                            "Error when trying to bipolar rereference data:\n"
                            "Some channel types of rereferenced channels have "
                            "not been specified, and they cannot be generated "
                            "based on the channels being rereferenced as they "
                            "are of different channel types."
                        )


    def _sort_ch_coords_new(self) -> None:
        """Resolves any missing entries for the channel coordinates of the new,
        rereferenced channels by taking the mean value of the coordinates of the
        channels being rereferenced.

        RAISES
        ------
        EntryLengthError
        -   Raised if any of the channel coordinates do not contain 3 entries,
            i.e. x, y, and z coordinates.
        """

        if self._ch_coords_new is None:
            self._ch_coords_new = [
                np.around(np.mean(
                    [self._ch_coords[self._ch_index[i][0]],
                    self._ch_coords[self._ch_index[i][1]]], axis=0
                ), 2) for i in range(self._n_channels)
            ]
        elif any(item is None for item in self._ch_types_new):
            for i, ch_coords in self._ch_coords_new:
                if ch_coords is None:
                    self._ch_coords_new[i] = np.around(np.mean(
                        [self._ch_coords[self._ch_index[i][0]],
                        self._ch_coords[self._ch_index[i][1]]], axis=0
                    ), 2)

        if not CheckLengthsList(self._ch_coords_new).equals_n(3):
            raise EntryLengthError(
                "Error when setting coordinates for the rereferenced data:\n"
                "Three, and only three coordinates (x, y, and z) must be "
                "present, but the rereferencing settings specify otherwise."
            )


    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.
        """

        self._check_input_lengths()
        self._sort_raw()
        self._sort_ch_names_new()
        self._sort_ch_types_new()
        self._sort_reref_types()
        self._sort_ch_coords_new()


    def _data_from_raw(self) -> None:
        """Extracts components of an mne.io.Raw object and returns them.
        -   Implemented in the parent class' method.
        """

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )


    def _raw_from_data(self) -> None:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.
        -   Implemented in the parent class' method.
        """

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.
        -   Implemented in the parent class' method.
        """

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types
        )


    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced, matching
        the format of the channel names used for rereferencing.
        """

        self._ch_index = deepcopy(self._ch_names_old)
        for sublist_i, sublist in enumerate(self._ch_names_old):
            for name_i, name in enumerate(sublist):
                self._ch_index[sublist_i][name_i] = self._ch_names.index(name)


    def _set_data(self) -> None:
        """Bipolar rereferences the data, subtracting one channel's data from
        another channel's data.
        """

        self._new_data = [
            self._data[self._ch_index[ch_i][0]]
            - self._data[self._ch_index[ch_i][1]]
            for ch_i in range(self._n_channels)
        ]


    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   If no coordinates are provided, the coordinates are calculated by
            taking the average coordinates of the anode and the cathode involved
            in each new, rereferenced channel.
        -   If some coordinates are provided, this calculation is performed for
            only those missing coordinates.

        RAISES
        ------
        EntryLengthError
        -   Raised if the provided list of coordinates for a channel does not
            have a length of 3 (i.e. an x, y, and z coordinate).
        """

        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if not CheckLengthsList(
                        self._ch_coords_new, [[]]
                    ).equals_n(3):
                        raise EntryLengthError(
                            "Error when setting coordinates for the "
                            "rereferenced data:\nThree, and only three "
                            "coordinates (x, y, and z) must be present, but "
                            "the rereferencing settings specify otherwise."
                        )
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set is False:
                self._new_ch_coords.append(
                    np.around(
                        np.mean(
                            [self._ch_coords[self._ch_index[ch_i][0]],
                            self._ch_coords[self._ch_index[ch_i][1]]], axis=0
                        ), 2
                    )
                )


    def _set_data_info(self) -> None:
        """Creates an mne.Info object containing information about the newly
        rereferenced data.
        -   Implemented in the parent class' method.
        """

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )


    def rereference(self) -> tuple[mne.io.Raw, list[str], dict[str, str]]:
        """Rereferences the data in an mne.io.Raw object.

        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object containing the bipolar rereferenced data.

        list[str]
        -   Names of the channels that were produced by the rereferencing.

        dict[str, str]
        -   Dictionary containing information about the type of rereferencing
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.
        """

        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self._ch_names_new, self.reref_types



class RerefCAR(Reref):
    """Common-average rereferences data in an mne.io.Raw object.
    -   Subclass of the abstract class Reref.

    PARAMETERS
    ----------
    raw : mne.io.Raw
    -   The mne.io.Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the mne.io.Raw object to rereference.

    ch_names_new : list[str or None] or None | default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   Missing values (None) will be set based on 'ch_names_old'.

    ch_types_new : list[str or None] or None | default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   Missing values (None) will be set based on the types of channels in
        'ch_names_old'.

    reref_types : list[str or None] or None | default None
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   Missing values (None) will be set as 'CAR', common-average
        rereferencing.

    ch_coords_new : list[list[int or float] or None] or None | default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]', the coordinates of the channels in
        'ch_names_old' in the mne.io.Raw object are used.
    -   If some sublists are '[]', those channels for which coordinates are
        given are used, whilst those channels for which the coordinates are
        missing have their coordinates taken from the mne.io.Raw object
        according to the corresponding channel in 'ch_names_old'.

    METHODS
    -------
    rereference
    -   Rereferences the data in an mne.io.Raw object.
    """

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list[str],
        ch_names_new: Optional[list[Optional[str]]] = None,
        ch_types_new: Optional[list[Optional[str]]] = None,
        reref_types: Optional[list[Optional[str]]] = None,
        ch_coords_new: Optional[list[Optional[list[realnum]]]] = None
        ) -> None:

        self.raw = raw
        self._data_from_raw()
        self._ch_names_old = ch_names_old
        self._index_old_channels()
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._reref_types = reref_types

        self._sort_inputs()


    def _check_input_lengths(self) -> None:
        """Checks that the lengths of the entries (representing the features of
        channels that will be rereferenced, e.g. channel names, coordinates,
        etc...) within a list are of the same length.
        -   This length corresponds to the number of channels in the
            rereferenced data.
        -   Implemented in the parent class' method.
        """

        self._n_channels = super()._check_input_lengths(
            [self._ch_names_old, self._ch_names_new, self._ch_types_new,
            self._reref_types, self._ch_coords_new]
        )


    def _sort_raw(self) -> None:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the parent class' method.
        """

        chs_to_analyse = np.unique(list(self._ch_names_old)).tolist()

        self.raw = super()._sort_raw(self.raw, chs_to_analyse)


    def _sort_ch_names_new(self) -> None:
        """Resolves any missing entries for the names of the new, rereferenced
        channels, taking names from the channels being rereferenced.
        """

        if self._ch_names_new is None:
            self._ch_names_new = list(self._ch_names_old)
        elif any(item is None for item in self._ch_names_new):
            for i, ch_name in enumerate(self._ch_names_new):
                if ch_name is None:
                    self._ch_names_new[i] = list(self._ch_names_old[i])

    def _sort_reref_types(self) -> None:
        """Resolves any missing entries from the rereference types of the new
        channels, setting them to 'CAR', common-average referencing.
        """

        if self._reref_types is None:
            self._reref_types = ['CAR' for i in range(self._n_channels)]
        elif any(item is None for item in self._reref_types):
            for i, reref_type in enumerate(self._reref_types):
                if reref_type is None:
                    self._reref_types[i] = 'CAR'


    def _sort_ch_types_new(self) -> None:
        """Resolves any missing entries from the channels types of the new
        channels, based on the types of channels they will be rereferenced from.
        """

        if self._ch_types_new is None:
            self._ch_types_new = list(
                self.raw.get_channel_types(self._ch_names_old)
            )
        elif any(item is None for item in self._ch_types_new):
            for i, ch_type in enumerate(self._ch_types_new):
                if ch_type is None:
                    self._ch_types_new[i] = list(
                        self.raw.get_channel_types(self._ch_names_old[i])
                    )


    def _sort_ch_coords_new(self) -> None:
        """Resolves any missing entries for the channel coordinates of the new,
        rereferenced channels by taking the value of the coordinates of the
        channels being rereferenced.

        RAISES
        ------
        EntryLengthError
        -   Raised if any of the channel coordinates do not contain 3 entries,
            i.e. x, y, and z coordinates.
        """

        if self._ch_coords_new is None:
            self._ch_coords_new = [
                self._ch_coords[self._ch_index[i]]
                for i in range(self._n_channels)
            ]
        elif any(item is None for item in self._ch_types_new):
            for i, ch_coords in self._ch_coords_new:
                if ch_coords is None:
                    self._ch_coords_new[i] = self._ch_coords[self._ch_index[i]]

        if not CheckLengthsList(self._ch_coords_new).equals_n(3):
            raise EntryLengthError(
                "Error when setting coordinates for the rereferenced data:\n"
                "Three, and only three coordinates (x, y, and z) must be "
                "present, but the rereferencing settings specify otherwise."
            )


    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.
        """

        self._check_input_lengths()
        self._sort_raw()
        self._sort_ch_names_new()
        self._sort_ch_types_new()
        self._sort_reref_types()
        self._sort_ch_coords_new()


    def _data_from_raw(self) -> None:
        """Extracts components of an mne.io.Raw object and returns them.
        -   Implemented in the parent class' method.
        """

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )


    def _raw_from_data(self) -> None:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.
        -   Implemented in the parent class' method.
        """

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.
        -   Implemented in the parent class' method.
        """

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types
        )


    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced, matching
        the format of the channel names used for rereferencing.
        """

        self._ch_index = (
            [self._ch_names.index(name) for name in self._ch_names_old]
        )


    def _set_data(self) -> None:
        """Common-average rereferences the data, subtracting the average of all
        channels' data from each individual channel.
        """

        avg_data = self._data[list(self._ch_index)].mean(axis=0)
        self._new_data = (
            [self._data[self._ch_index[ch_i]] - avg_data
            for ch_i in range(self._n_channels)]
        )


    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   If no coordinates are provided, the coordinates are calculated by
            taking the coordinates from the original channel involved in each
            new, rereferenced channel.
        -   If some coordinates are provided, this calculation is performed for
            only those missing coordinates.

        RAISES
        ------
        EntryLengthError
        -   Raised if the provided list of coordinates for a channel does not
            have a length of 3 (i.e. an x, y, and z coordinate).
        """

        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if not CheckLengthsList(
                        self._ch_coords_new, [[]]
                    ).equals_n(3):
                        raise Exception(
                            "Error when setting coordinates for the "
                            "rereferenced data.\nThree, and only three "
                            "coordinates (x, y, and z) must be present, but "
                            "the rereferencing settings specify otherwise."
                        )
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set is False:
                self._new_ch_coords.append(
                    self._ch_coords[self._ch_index[ch_i]]
                )


    def _set_data_info(self) -> None:
        """Creates an mne.Info object containing information about the newly
        rereferenced data.
        -   Implemented in the parent class' method.
        """

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )


    def rereference(self) -> tuple[mne.io.Raw, list[str], dict[str, str]]:
        """Rereferences the data in an mne.io.Raw object.

        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object containing the common-average rereferenced
            data.

        list[str]
        -   Names of the channels that were produced by the rereferencing.

        dict
        -   Dictionary containing information about the type of rereferencing
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.
        """

        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self._ch_names_new, self.reref_types



class RerefPseudo(Reref):
    """Pseudo rereferences data in an mne.io.Raw object.
    -   This allows e.g. rereferencing types to be assigned to the channels,
        channel coordinates to be set, etc... without any rereferencing
        occuring.
    -   This is useful if e.g. the channels were already hardware rereferenced.
    -   Subclass of the abstract class Reref.

    PARAMETERS
    ----------
    raw : mne.io.Raw
    -   The mne.io.Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the mne.io.Raw object to rereference.

    ch_names_new : list[str or None] or None | default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   Missing values (None) will be set based on 'ch_names_old'.

    ch_types_new : list[str or None] or None | default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   Missing values (None) will be set based on the types of channels in
        'ch_names_old'.

    reref_types : list[str]
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   No missing values (None) can be given, as the rereferencing type cannot
        be determined dynamically from this arbitrary rereferencing method.

    ch_coords_new : list[list[int or float] or None] or None | default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]', the coordinates of the channels in
        'ch_names_old' in the mne.io.Raw object are used.
    -   If some sublists are '[]', those channels for which coordinates are
        given are used, whilst those channels for which the coordinates are
        missing have their coordinates taken from the mne.io.Raw object
        according to the corresponding channel in 'ch_names_old'.

    METHODS
    -------
    rereference
    -   Rereferences the data in an mne.io.Raw object.
    """

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list[str],
        ch_names_new: Optional[list[Optional[str]]] = None,
        ch_types_new: Optional[list[Optional[str]]] = None,
        reref_types: Optional[list[Optional[str]]] = None,
        ch_coords_new: Optional[list[Optional[list[realnum]]]] = None
        ) -> None:

        self.raw = raw
        self._data_from_raw()
        self._ch_names_old = ch_names_old
        self._index_old_channels()
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._reref_types = reref_types

        self._sort_inputs()


    def _check_input_lengths(self) -> None:
        """Checks that the lengths of the entries (representing the features of
        channels that will be rereferenced, e.g. channel names, coordinates,
        etc...) within a list are of the same length.
        -   This length corresponds to the number of channels in the
            rereferenced data.
        -   Implemented in the parent class' method.
        """

        self._n_channels = super()._check_input_lengths(
            [self._ch_names_old, self._ch_names_new, self._ch_types_new,
            self._reref_types, self._ch_coords_new]
        )


    def _sort_raw(self) -> None:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the parent class' method.
        """

        chs_to_analyse = np.unique(list(self._ch_names_old)).tolist()

        self.raw = super()._sort_raw(self.raw, chs_to_analyse)


    def _sort_ch_names_new(self) -> None:
        """Resolves any missing entries for the names of the new, rereferenced
        channels, taking names from the channels being rereferenced.
        """

        if self._ch_names_new is None:
            self._ch_names_new = list(self._ch_names_old)
        elif any(item is None for item in self._ch_names_new):
            for i, ch_name in enumerate(self._ch_names_new):
                if ch_name is None:
                    self._ch_names_new[i] = list(self._ch_names_old[i])


    def _sort_reref_types(self) -> None:
        """Checks that all rereferencing types have been specified for the new
        channels, as these cannot be derived from the arbitrary method of pseudo
        rereferencing.

        RAISES
        ------
        TypeError
        -   Raised if the rereferencing type variable or any of its entries are
            of type None.
        """

        if None in self._reref_types:
            raise TypeError(
                "Error when pseudo rereferencing:\nRereferencing types of each "
                "new channel must be specified, there can be no missing entries"
                "."
            )


    def _sort_ch_types_new(self) -> None:
        """Resolves any missing entries from the channels types of the new
        channels, based on the types of channels they will be rereferenced from.
        """

        if self._ch_types_new is None:
            self._ch_types_new = list(
                self.raw.get_channel_types(self._ch_names_old)
            )
        elif any(item is None for item in self._ch_types_new):
            for i, ch_type in enumerate(self._ch_types_new):
                if ch_type is None:
                    self._ch_types_new[i] = list(
                        self.raw.get_channel_types(self._ch_names_old[i])
                    )


    def _sort_ch_coords_new(self) -> None:
        """Resolves any missing entries for the channel coordinates of the new,
        rereferenced channels by taking the value of the coordinates of the
        channels being rereferenced.

        RAISES
        ------
        EntryLengthError
        -   Raised if any of the channel coordinates do not contain 3 entries,
            i.e. x, y, and z coordinates.
        """

        if self._ch_coords_new is None:
            self._ch_coords_new = [
                self._ch_coords[self._ch_index[i]]
                for i in range(self._n_channels)
            ]
        elif any(item is None for item in self._ch_types_new):
            for i, ch_coords in self._ch_coords_new:
                if ch_coords is None:
                    self._ch_coords_new[i] = self._ch_coords[self._ch_index[i]]

        if not CheckLengthsList(self._ch_coords_new).equals_n(3):
            raise EntryLengthError(
                "Error when setting coordinates for the rereferenced data:\n"
                "Three, and only three coordinates (x, y, and z) must be "
                "present, but the rereferencing settings specify otherwise."
            )


    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.
        """

        self._check_input_lengths()
        self._sort_raw()
        self._sort_ch_names_new()
        self._sort_ch_types_new()
        self._sort_reref_types()
        self._sort_ch_coords_new()


    def _data_from_raw(self) -> None:
        """Extracts components of an mne.io.Raw object and returns them.
        -   Implemented in the parent class' method.
        """

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )


    def _raw_from_data(self) -> None:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.
        -   Implemented in the parent class' method.
        """

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.
        -   Implemented in the parent class' method.
        """

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types
        )


    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced, matching
        the format of the channel names used for rereferencing.
        """

        self._ch_index = (
            [self._ch_names.index(name) for name in self._ch_names_old]
        )


    def _set_data(self) -> None:
        """Pseudo rereferences the data, setting the data for each new,
        rereferenced channel equal to that of the corresponding channel in the
        original data.
        """

        self._new_data = (
            [self._data[self._ch_index[ch_i]]
            for ch_i in range(self._n_channels)]
        )


    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   If no coordinates are provided, the coordinates are calculated by
            taking the coordinates from the original channel involved in each
            new, rereferenced channel.
        -   If some coordinates are provided, this calculation is performed for
            only those missing coordinates.

        RAISES
        ------
        EntryLengthError
        -   Raised if the provided list of coordinates for a channel does not
            have a length of 3 (i.e. an x, y, and z coordinate).
        """

        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if not CheckLengthsList(
                        self._ch_coords_new, [[]]
                    ).equals_n(3):
                        raise Exception(
                            "Error when setting coordinates for the "
                            "rereferenced data:\nThree, and only three "
                            "coordinates (x, y, and z) must be present, but "
                            "the rereferencing settings specify otherwise."
                        )
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set is False:
                self._new_ch_coords.append(
                    np.around(self._ch_coords[self._ch_index[ch_i]], 2)
                )


    def _set_data_info(self) -> None:
        """Creates an mne.Info object containing information about the newly
        rereferenced data.
        -   Implemented in the parent class' method.
        """

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )


    def rereference(self) -> tuple[mne.io.Raw, list[str], dict[str, str]]:
        """Rereferences the data in an mne.io.Raw object.

        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object containing the pseudo rereferenced
            data.

        list[str]
        -   Names of the channels that were produced by the rereferencing.

        dict[str, str]
        -   Dictionary containing information about the type of rereferencing
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.
        """

        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self._ch_names_new, self.reref_types
