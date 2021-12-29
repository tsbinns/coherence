from abc import ABC, abstractmethod
import numpy as np
import mne

from coh_check_entries import CheckLengthsDict, CheckLengthsList




class Reref(ABC):

    @abstractmethod
    def _data_from_raw(self,
        raw
        ) -> tuple[np.array, mne.Info, list]:
        
        return raw.get_data(reject_by_annotation='omit').copy(), raw.info.copy(), raw.info['ch_names'].copy(), raw._get_channel_positions().copy().tolist()


    @abstractmethod
    def _raw_from_data(self,
        data,
        data_info,
        ch_coords
        ) -> mne.io.Raw:
        
        raw = mne.io.RawArray(data, data_info)
        raw._set_channel_positions(ch_coords, data_info['ch_names'])

        return raw


    @abstractmethod
    def _is_empty(self,
        check_entry,
        empty = []
        ) -> bool:

        if check_entry == empty:
            is_empty = True
        else:
            is_empty = False

        return is_empty


    @abstractmethod
    def _handle_empty(self,
        new_values,
        old_values,
        empty = []
        ):

        if self._is_empty(new_values, empty):
            new_values = old_values

        return new_values


    @abstractmethod
    def _store_rereference_types(self,
        settings: dict
        ) -> dict:

        ch_names = self._handle_empty(settings['ch_names_old'], settings['ch_names_new'], [])
        reref_types = settings['reref_types']

        equal_lengths, n_channels = CheckLengthsList.check_identical([ch_names, reref_types])
        if not equal_lengths:
            raise Exception(f"Error when trying to assign channel rereferencing types.\nThere are {len(ch_names)} channels, but {len(reref_types)} rereferencing types.")

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
    def _set_data_info():
        pass


    @abstractmethod
    def _rereference():
        pass



class RerefBipolar(Reref):

    def __init__(self,
        raw: mne.io.Raw,
        settings: dict
        ):

        equal_lengths, self._n_channels = CheckLengthsDict.check_identical(settings, ignore_values=[[]])
        if not equal_lengths:
            raise Exception(f"Error when reading rereferencing settings.\nThe length of entries within the settings dictionary are not identical:\n{self._n_channels}")

        self.raw = raw
        self._settings = settings

        return self._rereference()


    def _data_from_raw(self):

        self._data, self._data_info, self._ch_names, self._ch_coordinates = super()._data_from_raw(self.raw)

    
    def _raw_from_data(self):

        self.raw = super()._raw_from_data(self._data, self._data_info, self._ch_coordinates)


    def _is_empty(self,
        check_entry,
        empty = []
        ) -> bool:

        return super()._is_empty(check_entry, empty)


    def _handle_empty(self,
        new_values,
        old_values,
        empty = []
        ):

        return super()._handle_empty(new_values, old_values, empty)


    def _store_rereference_types(self):

        self.reref_types = super()._store_rereference_types(self._settings)


    def _index_old_channels(self):
        
        self._ch_index = self._settings['ch_names_old'].copy()
        for sublist_i, sublist in enumerate(self._settings['ch_names_old']):
            for name_i, name in enumerate(sublist):
                self._ch_index[sublist_i][name_i] = self._ch_names.index(name)


    def _set_data(self):
        
        if CheckLengthsList.check_equals_n(self._settings['ch_names_old'], 2):
            raise Exception(f"Error when bipolar rereferencing data.\nThis must involve two, and only two channels of data, but the rereferencing settings specify otherwise.")
        
        self._new_data = []
        for ch_i in range(self._n_channels):
            self._new_data.append(self._data[self._ch_index[ch_i][0]] - self._data[self._ch_index[ch_i][1]])


    def _set_coordinates(self):
        
        self._new_ch_coordinates = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._settings['ch_coords'] != []:
                if self._settings['ch_coords'][ch_i] != []:
                    if CheckLengthsList.check_equals_n(self._settings['ch_coords'], 3):
                        raise Exception(f"Error when setting coordinates for the rereferenced data.\nThree, and only three coordinates (x, y, and z) must be present, but the rereferencing settings specify otherwise.")
                    self._new_ch_coordinates.append(self._settings['ch_coords'][ch_i])
                    coords_set = True
            if coords_set == False:
                self._new_ch_coordinates.append(
                    np.around(np.mean([self._ch_coordinates[self._ch_index[ch_i][0]],
                        self._ch_coordinates[self._ch_index[ch_i][1]]], axis=0), 2)
                    )


    def _set_data_info(self):

        self._new_data_info = mne.create_info(
            self._settings['ch_names_new'], self._data_info['sfreq'], self._settings['ch_types']
            )
        

    def _rereference(self
        ) -> tuple[mne.io.Raw, dict]:

        self._data_from_raw()
        self._set_data()
        self._set_coordinates()
        self._set_data_info()

        return self._raw_from_data(), self._store_rereference_types()