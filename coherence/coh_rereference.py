from abc import ABC, abstractmethod
import numpy as np
import mne

from coh_check_entries import CheckLengthsDict, CheckLengthsList




class Reref(ABC):

    @abstractmethod
    def _data_from_raw(self,
        raw
        ) -> tuple[np.array, mne.Info, list]:
        
        return raw.get_data(reject_by_annotation='omit').copy(), raw.info.copy(), raw._get_channel_positions().copy().tolist()


    @abstractmethod
    def _raw_from_data(self,
        data,
        data_info,
        ch_coordinates
        ) -> tuple[mne.io.Raw, list]:
        pass


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

        if len(ch_names) != len(reref_types):
            raise Exception(f"Error when trying to assign channel rereferencing types.\nThere are {len(ch_names)} channels, but {len(reref_types)} rereferencing types.")
        else:
            len_entries = len(ch_names)

        return {ch_names[i]: reref_types[i] for i in range(len_entries)}
        

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
    def rereference():
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


    def _data_from_raw(self):

        self._data, self._data_info, self._ch_coordinates = super()._data_from_raw(self.raw)

    
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


    def _set_data(self):
        
        if CheckLengthsList.check_equals_n(self._settings['ch_names_old'], 2):
            raise Exception(f"Error when bipolar rereferencing data.\nThis must involve two, and only two channels of data, but the rereferencing settings specify otherwise.")
        
        for ch_i in len(self._n_channels):



    def rereference(self
        ) -> tuple[mne.io.Raw, dict]:

        self.data_from_raw()
        self._set_data() # not implemented
        self._set_coordinates() # not implemented
        self._set_data_info() # not implemented

        return self._raw_from_data(), self._store_rereference_types()