import mne
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

from coh_check_entries import CheckLengthsList
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
            lengths_to_check, [[]]
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
    def _sort_inputs():
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.
        -   Implemented in the subclasses' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        pass

    @abstractmethod
    def _data_from_raw(self,
        raw: mne.io.Raw
        ) -> tuple[
            np.array, mne.Info, list[str], list[list[Union[int, float]]]
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
        data: np.array,
        data_info: mne.Info,
        ch_coords: list = []
        ) -> mne.io.Raw:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.

        PARAMETERS
        ----------
        data : np.array
        -   Array of the rereferenced data with shape
            [n_channels x n_timepoints].

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
    def _index_old_channels():
        """Creates an index of channels that are being rereferenced.
        -   Implemented in the subclasses' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        pass


    @abstractmethod
    def _set_data():
        """Rereferences the data.
        -   Implemented in the subclasses' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        pass


    @abstractmethod
    def _set_coordinates():
        """Sets the coordinates of the new, rereferenced channels.
        -   Implemented in the subclasses.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        pass


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
    def rereference():
        """Rereferences the data in an mne.io.Raw object.
        -   Implemented in the subclasses' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        pass



class RerefBipolar(Reref):
    """Bipolar rereferences data in an mne.io.Raw object.
    -   Subclass of the abstract class Reref.

    PARAMETERS
    ----------
    raw : mne.io.Raw
    -   The mne.io.Raw object containing the data to be rereferenced.

    ch_names_old : list[list[str]]
    -   The names of the channels in the mne.io.Raw object to rereference.
    -   Each entry of the list should be a list of two channel names (i.e. a 
        cathode and an anode).

    ch_names_new : list[str]
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.

    ch_types_new : list[str]
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.

    reref_types : list[str]
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.

    ch_coords_new : empty list or list[empty list or list[int or float]]
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
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: list[str],
        reref_types: list[str],
        ch_coords_new: list = []
        ) -> None:

        self.raw = raw
        self._ch_names_old = ch_names_old
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

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._n_channels = super()._check_input_lengths(
            [self._ch_names_old, self._ch_names_new, self._ch_types_new, 
            self._reref_types, self._ch_coords_new]
        )

        
    def _sort_raw(self) -> None:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        chs_to_analyse = np.unique(
            [name for names in self._ch_names_old for name in names]
        ).tolist()

        self.raw = super()._sort_raw(self.raw, chs_to_analyse)


    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._check_input_lengths()
        self._sort_raw()


    def _data_from_raw(self) -> None:
        """Extracts components of an mne.io.Raw object and returns them.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )

    
    def _raw_from_data(self) -> None:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types
        )


    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced, matching
        the format of the channel names used for rereferencing.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        
        self._ch_index = deepcopy(self._ch_names_old)
        for sublist_i, sublist in enumerate(self._ch_names_old):
            for name_i, name in enumerate(sublist):
                self._ch_index[sublist_i][name_i] = self._ch_names.index(name)


    def _set_data(self) -> None:
        """Bipolar rereferences the data, subtracting one channel's data from
        another channel's data.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A

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
        
        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A

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
            if coords_set == False:
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

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )
        

    def rereference(self) -> tuple[mne.io.Raw, dict[str, str]]:
        """Rereferences the data in an mne.io.Raw object.
        
        PARAMETERS
        ----------
        N/A
        
        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object containing the bipolar rereferenced data.
        
        dict
        -   Dictionary containing information about the type of rereferencing 
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.
        """

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self.reref_types



class RerefCAR(Reref):
    """Common-average rereferences data in an mne.io.Raw object.
    -   Subclass of the abstract class Reref.

    PARAMETERS
    ----------
    raw : mne.io.Raw
    -   The mne.io.Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the mne.io.Raw object to rereference.

    ch_names_new : list[str]
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.

    ch_types_new : list[str]
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.

    reref_types : list[str]
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.

    ch_coords_new : empty list or list[empty list or list[int or float]]
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
        ch_names_new: list[str],
        ch_types_new: list[str],
        reref_types: list[str],
        ch_coords_new: list = []
        ) -> None:

        self.raw = raw
        self._ch_names_old = ch_names_old
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

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._n_channels = super()._check_input_lengths(
            [self._ch_names_old, self._ch_names_new, self._ch_types_new, 
            self._reref_types, self._ch_coords_new]
        )

        
    def _sort_raw(self) -> None:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        chs_to_analyse = np.unique(
            [name for name in self._ch_names_old]
        ).tolist()

        self.raw = super()._sort_raw(self.raw, chs_to_analyse)


    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._check_input_lengths()
        self._sort_raw()


    def _data_from_raw(self) -> None:
        """Extracts components of an mne.io.Raw object and returns them.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )

    
    def _raw_from_data(self) -> None:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types
        )


    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced, matching
        the format of the channel names used for rereferencing.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        
        self._ch_index = (
            [self._ch_names.index(name) for name in self._ch_names_old]
        )


    def _set_data(self) -> None:
        """Common-average rereferences the data, subtracting the average of all
        channels' data from each individual channel.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        
        avg_data = self._data[[ch_i for ch_i in self._ch_index]].mean(axis=0)
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
        
        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A

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
            if coords_set == False:
                self._new_ch_coords.append(
                    self._ch_coords[self._ch_index[ch_i]]
                )


    def _set_data_info(self) -> None:
        """Creates an mne.Info object containing information about the newly
        rereferenced data.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )
        

    def rereference(self) -> tuple[mne.io.Raw, dict[str, str]]:
        """Rereferences the data in an mne.io.Raw object.
        
        PARAMETERS
        ----------
        N/A
        
        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object containing the common-average rereferenced
            data.
        
        dict
        -   Dictionary containing information about the type of rereferencing 
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.
        """

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self.reref_types



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

    ch_names_new : list[str]
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.

    ch_types_new : list[str]
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.

    reref_types : list[str]
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.

    ch_coords_new : empty list or list[empty list or list[int or float]]
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
        ch_names_new: list[str],
        ch_types_new: list[str],
        reref_types: list[str],
        ch_coords_new: list = []
        ) -> None:

        self.raw = raw
        self._ch_names_old = ch_names_old
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

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._n_channels = super()._check_input_lengths(
            [self._ch_names_old, self._ch_names_new, self._ch_types_new, 
            self._reref_types, self._ch_coords_new]
        )

        
    def _sort_raw(self) -> None:
        """Drops channels irrelevant to the rereferencing from an mne.io.Raw
        object.
        -   Partially implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        chs_to_analyse = np.unique(
            [name for name in self._ch_names_old]
        ).tolist()

        self.raw = super()._sort_raw(self.raw, chs_to_analyse)


    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._check_input_lengths()
        self._sort_raw()


    def _data_from_raw(self) -> None:
        """Extracts components of an mne.io.Raw object and returns them.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )

    
    def _raw_from_data(self) -> None:
        """Generates an mne.io.Raw object based on the rereferenced data and its
        associated information.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:
        """Generates a dictionary of key:value pairs consisting of channel name
        : rereferencing type.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types
        )


    def _index_old_channels(self) -> None:
        """Creates an index of channels that are being rereferenced, matching
        the format of the channel names used for rereferencing.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        
        self._ch_index = (
            [self._ch_names.index(name) for name in self._ch_names_old]
        )


    def _set_data(self) -> None:
        """Pseudo rereferences the data, setting the data for each new,
        rereferenced channel equal to that of the corresponding channel in the
        original data.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
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
        
        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A

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
            if coords_set == False:
                self._new_ch_coords.append(
                    np.around(self._ch_coords[self._ch_index[ch_i]], 2)
                )


    def _set_data_info(self) -> None:
        """Creates an mne.Info object containing information about the newly
        rereferenced data.
        -   Implemented in the parent class' method.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )
        

    def rereference(self) -> tuple[mne.io.Raw, dict[str, str]]:
        """Rereferences the data in an mne.io.Raw object.
        
        PARAMETERS
        ----------
        N/A
        
        RETURNS
        -------
        mne.io.Raw
        -   The mne.io.Raw object containing the common-average rereferenced
            data.
        
        dict
        -   Dictionary containing information about the type of rereferencing 
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.
        """

        
        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self.reref_types


