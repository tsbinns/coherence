from typing import Optional
import mne
import numpy as np

from coh_check_entries import CheckLengthsList
from coh_dtypes import realnum
from coh_exceptions import EntryLengthError

from coh_rereference import Reref




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
