import mne
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

from coh_check_entries import CheckLengthsList




class Reref(ABC):
    """Abstract class for rereferencing data in mne.io.Raw objects.

    METHODS
    -------
    rereference (abstract)
    -   Rereferences the data in an mne.io.Raw object.

    SUBCLASSES
    ----------
    RerefBipolar
    -   Bipolar rereferences data.

    RerefCAR
    -   Common-average rereferences data.

    RerefPseudo
    -   Psuedo rereferences data. This allows you to alter characteristics of
        the mne.io.Raw object (e.g. channel coordinates) and assign a
        rereferencing type to the channels without altering the data. This is
        useful if e.g. the channels were already hardware rereferenced.
    """

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
        """Creates an index of channels that are being rereferenced. Implemented
        in the subclasses.

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
        """Rereferences the data. Implemented in the subclasses.

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
        """Sets the coordinates of the newly rereferenced channels. Implemented
        in the subclasses.

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
        """Rereferences the data in an mne.io.Raw object. Implemented in the
        subclasses.

        PARAMETERS
        ----------
        N/A

        RETURNS
        -------
        N/A
        """
        pass



class RerefBipolar(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
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

        lengths_to_check = [
            self._ch_names_old, self._ch_names_new, self._ch_types_new, 
            self._reref_types
        ]
        if self._ch_coords_new != []:
            lengths_to_check.append(self._ch_coords_new)

        equal_lengths, self._n_channels = CheckLengthsList(
            lengths_to_check, [[]]
        ).identical()

        if not equal_lengths:
            raise Exception(
                "Error when reading rereferencing settings:\nThe length of "
                "entries within the settings dictionary are not identical:\n"
                f"{self._n_channels}"
            )

        
    def _sort_raw(self) -> None:

        chs_to_analyse = np.unique(
            [name for names in self._ch_names_old for name in names]
        ).tolist()

        self.raw.drop_channels(
            [name for name in self.raw.info['ch_names']
            if name not in chs_to_analyse]
        )

        self.raw.reorder_channels(chs_to_analyse)


    def _sort_inputs(self) -> None:

        self._check_input_lengths()
        self._sort_raw()


    def _data_from_raw(self) -> None:

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )

    
    def _raw_from_data(self) -> None:

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types, self._n_channels
        )


    def _index_old_channels(self) -> None:
        
        self._ch_index = deepcopy(self._ch_names_old)
        for sublist_i, sublist in enumerate(self._ch_names_old):
            for name_i, name in enumerate(sublist):
                self._ch_index[sublist_i][name_i] = self._ch_names.index(name)


    def _set_data(self) -> None:
        
        if not CheckLengthsList(self._ch_names_old).equals_n(2):
            raise Exception(
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
                    np.around(
                        np.mean(
                            [self._ch_coords[self._ch_index[ch_i][0]],
                            self._ch_coords[self._ch_index[ch_i][1]]], axis=0
                        ), 2
                    )
                )


    def _set_data_info(self) -> None:

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )
        

    def rereference(self) -> tuple[mne.io.Raw, dict]:

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self.reref_types



class RerefCAR(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
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

        lengths_to_check = [
            self._ch_names_old, self._ch_names_new, self._ch_types_new, 
            self._reref_types
        ]
        if self._ch_coords_new != []:
            lengths_to_check.append(self._ch_coords_new)

        equal_lengths, self._n_channels = CheckLengthsList(
            lengths_to_check, [[]]
        ).identical()

        if not equal_lengths:
            raise Exception(
                "Error when reading rereferencing settings:\nThe length of "
                "entries within the settings dictionary are not identical:\n"
                f"{self._n_channels}"
            )

        
    def _sort_raw(self) -> None:

        chs_to_analyse = np.unique(
            [name for name in self._ch_names_old]
        ).tolist()

        self.raw.drop_channels(
            [name for name in self.raw.info['ch_names']
            if name not in chs_to_analyse]
        )

        self.raw.reorder_channels(chs_to_analyse)


    def _sort_inputs(self) -> None:

        self._check_input_lengths()
        self._sort_raw()


    def _data_from_raw(self) -> None:

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )

    
    def _raw_from_data(self) -> None:

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types, self._n_channels
        )


    def _index_old_channels(self) -> None:
        
        self._ch_index = (
            [self._ch_names.index(name) for name in self._ch_names_old]
        )


    def _set_data(self) -> None:
        
        avg_data = self._data[[ch_i for ch_i in self._ch_index]].mean(axis=0)
        self._new_data = (
            [self._data[self._ch_index[ch_i]] - avg_data
            for ch_i in range(self._n_channels)]
        )


    def _set_coordinates(self) -> None:
        
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

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )
        

    def rereference(self) -> tuple[mne.io.Raw, dict]:

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self.reref_types



class RerefPseudo(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
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

        lengths_to_check = [
            self._ch_names_old, self._ch_names_new, self._ch_types_new, 
            self._reref_types
        ]
        if self._ch_coords_new != []:
            lengths_to_check.append(self._ch_coords_new)

        equal_lengths, self._n_channels = CheckLengthsList(
            lengths_to_check, [[]]
        ).identical()

        if not equal_lengths:
            raise Exception(
                "Error when reading rereferencing settings:\nThe length of "
                "entries within the settings dictionary are not identical:\n"
                f"{self._n_channels}"
            )

        
    def _sort_raw(self) -> None:

        chs_to_analyse = np.unique(
            [name for name in self._ch_names_old]
        ).tolist()

        self.raw.drop_channels(
            [name for name in self.raw.info['ch_names']
            if name not in chs_to_analyse]
        )

        self.raw.reorder_channels(chs_to_analyse)


    def _sort_inputs(self) -> None:

        self._check_input_lengths()
        self._sort_raw()


    def _data_from_raw(self) -> None:

        self._data, self._data_info, self._ch_names, self._ch_coords = (
            super()._data_from_raw(self.raw)
        )

    
    def _raw_from_data(self) -> None:

        self.raw = super()._raw_from_data(
            self._new_data, self._new_data_info, self._new_ch_coords
        )


    def _store_rereference_types(self) -> None:

        self.reref_types = super()._store_rereference_types(
            self._ch_names_new, self._reref_types, self._n_channels
        )


    def _index_old_channels(self) -> None:
        
        self._ch_index = (
            [self._ch_names.index(name) for name in self._ch_names_old]
        )


    def _set_data(self) -> None:
        
        self._new_data = (
            [self._data[self._ch_index[ch_i]]
            for ch_i in range(self._n_channels)]
        )


    def _set_coordinates(self) -> None:
        
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

        self._new_data_info = super()._set_data_info(
            self._ch_names_new, self._ch_types_new, self._data_info
        )
        

    def rereference(self) -> tuple[mne.io.Raw, dict]:

        
        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_rereference_types()

        return self.raw, self.reref_types


