from abc import ABC, abstractmethod
import numpy as np
import mne

from coh_check_entries import CheckLengthsDict, CheckLengthsList




class Reref(ABC):

    @abstractmethod
    def _data_from_raw(self,
        raw: mne.io.Raw
        ) -> tuple[np.array, mne.Info, list, list]:
        
        return (raw.get_data(reject_by_annotation='omit').copy(),
                raw.info.copy(),
                raw.info['ch_names'].copy(),
                raw._get_channel_positions().copy().tolist())


    @abstractmethod
    def _raw_from_data(self,
        data: np.array,
        data_info: mne.Info,
        ch_coords: list
        ) -> mne.io.Raw:
        
        raw = mne.io.RawArray(data, data_info)
        raw._set_channel_positions(ch_coords, data_info['ch_names'])

        return raw


    @abstractmethod
    def _store_rereference_types(self,
        ch_names: list,
        reref_types: list,
        n_channels: int
        ) -> dict:

        return {ch_names[i]: reref_types[i] for i in range(n_channels)}
        

    @abstractmethod
    def _index_old_channels():
        pass


    @abstractmethod
    def _set_data():
        pass


    @abstractmethod
    def _set_coordinates():
        pass


    @abstractmethod
    def _set_data_info(self,
        ch_names: list,
        sfreq: int,
        ch_types: list
        ) -> mne.Info:

        return mne.create_info(ch_names, sfreq, ch_types)


    @abstractmethod
    def _rereference():
        pass



class RerefBipolar(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list = [],
        ) -> None:

        lengths_to_check = [ch_names_old, ch_names_new, ch_types_new, reref_types]
        if ch_coords_new != []:
            lengths_to_check.append(ch_coords_new)
        equal_lengths, self._n_channels = CheckLengthsList.check_identical(lengths_to_check)
        if not equal_lengths:
            raise Exception(f"Error when reading rereferencing settings.\nThe length of entries within the settings dictionary are not identical:\n{self._n_channels}")

        self.raw = raw
        self._ch_names_old = ch_names_old
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._reref_types = reref_types


    def _data_from_raw(self
        ) -> None:

        self._data, self._data_info, self._ch_names, self._ch_coords = super()._data_from_raw(self.raw)

    
    def _raw_from_data(self
        ) -> None:

        self.raw = super()._raw_from_data(self._data, self._data_info, self._ch_coords)


    def _store_rereference_types(self
        ) -> None:

        self.reref_types = super()._store_rereference_types(self._ch_names_new, self._reref_types, self._n_channels)


    def _index_old_channels(self
        ) -> None:
        
        self._ch_index = self._ch_names_old.copy()
        for sublist_i, sublist in enumerate(self._ch_names_old):
            for name_i, name in enumerate(sublist):
                self._ch_index[sublist_i][name_i] = self._ch_names.index(name)


    def _set_data(self
        ) -> None:
        
        if not CheckLengthsList.check_equals_n(self._ch_names_old, 2):
            raise Exception(f"Error when bipolar rereferencing data.\nThis must involve two, and only two channels of data, but the rereferencing settings specify otherwise.")
        
        self._new_data = [self._data[self._ch_index[ch_i][0]] - self._data[self._ch_index[ch_i][1]] for ch_i in
            range(self._n_channels)]


    def _set_coordinates(self
        ) -> None:
        
        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if CheckLengthsList.check_equals_n(self._ch_coords_new, 3):
                        raise Exception(f"Error when setting coordinates for the rereferenced data.\nThree, and only three coordinates (x, y, and z) must be present, but the rereferencing settings specify otherwise.")
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set == False:
                self._new_ch_coords.append(
                    np.around(np.mean([self._ch_coords[self._ch_index[ch_i][0]],
                        self._ch_coords[self._ch_index[ch_i][1]]], axis=0), 2)
                    )


    def _set_data_info(self
        ) -> None:

        self._new_data_info = super()._set_data_info(self._ch_names_new, self._data_info['sfreq'], self._ch_types)
        

    def rereference(self
        ) -> tuple[mne.io.Raw, dict]:

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()

        return self._raw_from_data(), self._store_rereference_types()



class RerefCAR(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list = [],
        ) -> None:

        lengths_to_check = [ch_names_old, ch_names_new, ch_types_new, reref_types]
        if ch_coords_new != []:
            lengths_to_check.append(ch_coords_new)
        equal_lengths, self._n_channels = CheckLengthsList.check_identical(lengths_to_check)
        if not equal_lengths:
            raise Exception(f"Error when reading rereferencing settings.\nThe length of entries within the settings dictionary are not identical:\n{self._n_channels}")

        self.raw = raw
        self._ch_names_old = ch_names_old
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._reref_types = reref_types


    def _data_from_raw(self
        ) -> None:

        self._data, self._data_info, self._ch_names, self._ch_coords = super()._data_from_raw(self.raw)

    
    def _raw_from_data(self
        ) -> None:

        self.raw = super()._raw_from_data(self._data, self._data_info, self._ch_coords)


    def _store_rereference_types(self
        ) -> None:

        self.reref_types = super()._store_rereference_types(self._ch_names_new, self._reref_types, self._n_channels)


    def _index_old_channels(self
        ) -> None:
        
        self._ch_index = [self._ch_names.index(name) for name in self._ch_names_old]


    def _set_data(self
        ) -> None:
        
        avg_data = self._data[[ch_i for ch_i in self._ch_index]].mean(axis=0)
        self._new_data = [self._data[self._ch_index[ch_i]] - avg_data for ch_i in range(self._n_channels)]


    def _set_coordinates(self
        ) -> None:
        
        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if CheckLengthsList.check_equals_n(self._ch_coords_new, 3):
                        raise Exception(f"Error when setting coordinates for the rereferenced data.\nThree, and only three coordinates (x, y, and z) must be present, but the rereferencing settings specify otherwise.")
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set == False:
                self._new_ch_coords.append(self._ch_coords[self._ch_index[ch_i]])


    def _set_data_info(self
        ) -> None:

        self._new_data_info = super()._set_data_info(self._ch_names_new, self._data_info['sfreq'], self._ch_types)
        

    def rereference(self
        ) -> tuple[mne.io.Raw, dict]:

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()

        return self._raw_from_data(), self._store_rereference_types()



class RerefPseudo(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list = [],
        ) -> None:

        lengths_to_check = [ch_names_old, ch_names_new, ch_types_new, reref_types]
        if ch_coords_new != []:
            lengths_to_check.append(ch_coords_new)
        equal_lengths, self._n_channels = CheckLengthsList.check_identical(lengths_to_check)
        if not equal_lengths:
            raise Exception(f"Error when reading rereferencing settings.\nThe length of entries within the settings dictionary are not identical:\n{self._n_channels}")

        self.raw = raw
        self._ch_names_old = ch_names_old
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._reref_types = reref_types


    def _data_from_raw(self
        ) -> None:

        self._data, self._data_info, self._ch_names, self._ch_coords = super()._data_from_raw(self.raw)

    
    def _raw_from_data(self
        ) -> None:

        self.raw = super()._raw_from_data(self._data, self._data_info, self._ch_coords)


    def _store_rereference_types(self
        ) -> None:

        self.reref_types = super()._store_rereference_types(self._ch_names_new, self._reref_types, self._n_channels)


    def _index_old_channels(self
        ) -> None:
        
        self._ch_index = [self._ch_names.index(name) for name in self._ch_names_old]


    def _set_data(self
        ) -> None:
        
        self._new_data = [self._data[self._ch_index[ch_i]] for ch_i in range(self._n_channels)]


    def _set_coordinates(self
        ) -> None:
        
        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if CheckLengthsList.check_equals_n(self._ch_coords_new, 3):
                        raise Exception(f"Error when setting coordinates for the rereferenced data.\nThree, and only three coordinates (x, y, and z) must be present, but the rereferencing settings specify otherwise.")
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set == False:
                self._new_ch_coords.append(self._ch_coords[self._ch_index[ch_i]])


    def _set_data_info(self
        ) -> None:

        self._new_data_info = super()._set_data_info(self._ch_names_new, self._data_info['sfreq'], self._ch_types)
        

    def rereference(self
        ) -> tuple[mne.io.Raw, dict]:

        self._data_from_raw()
        self._index_old_channels()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()

        return self._raw_from_data(), self._store_rereference_types()