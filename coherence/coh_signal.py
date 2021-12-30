import mne_bids
import numpy as np
import mne
from coh_rereference import Reref, RerefBipolar, RerefCAR, RerefPseudo




class Signal:

    def __init__(self,
        verbose: bool = True
        ) -> None:

        self._verbose = verbose
        self.processing_steps = []
        self.extra_info = {}


    def get_coordinates(self
        ) -> list:

        return self.raw._get_channel_positions().copy().tolist()


    def set_coordinates(self,
        ch_coords: list,
        ch_names: list
        ) -> None:

        self.raw._set_channel_positions(ch_coords, ch_names)

        if self._verbose:
            print(f"Setting channel coordinates to:\n{ch_coords}.")


    def get_data(self
        ) -> np.array:

        return self.raw.get_data(reject_by_annotation='omit').copy()


    def load_raw(self,
        path_raw: mne_bids.BIDSPath
        ) -> None:

        self._path_raw = path_raw
        self.raw = mne_bids.read_raw_bids(bids_path=self.path_raw, verbose=False)
        self.raw.load_data()

        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.")

    
    def load_annotations(self,
        path_annots: str
        ) -> None:

        self.raw.set_annotations(mne.read_annotations(path_annots))

        if self._verbose:
            print(f"Applying annotations to the data from the filepath:\n{path_annots}.")

        self._annotations_added = True
        self.processing_steps.append(['annotations_added', True])


    def _pick_extra_info(self,
        ch_names: list
        ) -> None:

        for key in self.extra_info.keys():
            new_entry = {ch_name: self.extra_info[key][ch_name] for ch_name in ch_names}
            self.extra_info[key] = new_entry

    
    def pick_channels(self,
        ch_names: list
        ) -> None:

        self.raw.pick_channels(ch_names)
        self._pick_extra_info(ch_names)

        if self._verbose:
            print(f"Picking specified channels from the data.\nChannels: {ch_names}.")

        self.processing_steps.append(['channel_picks', ch_names])

    
    def bandpass_filter(self,
        freqs: list
        ) -> None:

        if len(freqs) != 2:
            raise Exception(f"Error when bandpass filtering the data.\nTwo frequencies are required, but {len(freqs)} are given.")

        self.raw.filter(freqs[0], freqs[1])

        if self._verbose:
            print(f"Bandpass filtering the data.\nLow frequency: {freqs[0]} Hz. High frequency: {freqs[1]} Hz.")

        self._bandpass_filtered = True
        self.processing_steps.append(['bandpass_filter', freqs])

    
    def notch_filter(self,
        line_freq: int
        ) -> None:

        freqs = np.arange(0, self.raw.info['lowpass']+1, line_freq)
        self.raw.notch_filter(freqs)

        if self._verbose:
            print(f"Notch filtering the data with line noise frequency {line_freq} Hz at the following frequencies (Hz): {freqs}.")

        self._notch_filtered = True
        self.processing_steps.append(['notch_filter', freqs])


    def resample(self,
        resample_freq: int
        ) -> None:

        self.raw.resample(resample_freq)

        if self._verbose:
            print(f"Resampling the data at {resample_freq} Hz.")

        self._resampled = True
        self.processing_steps(['resample', resample_freq])


    def _apply_rereference(self,
        RerefMethod: Reref,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list
        ) -> list:

        RerefObject = RerefMethod(
            self.raw,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            reref_types,
            ch_coords_new
            )

        return RerefObject.rereference()


    def _check_conflicting_channels(self,
        ch_names_1: list,
        ch_names_2: list
        ) -> list:

        return [name for name in ch_names_1 if name in ch_names_2]


    def _remove_conflicting_channels(self,
        raw_to_alter: mne.io.Raw,
        ch_to_remove: list
        ) -> None:

        raw_to_alter.drop_channels(ch_to_remove)
        

    def _append_rereferenced_raw(self,
        rerefed_raw: mne.io.Raw
        ) -> None:

        conflicting_channels = self._check_conflicting_channels(self.raw.info['ch_names'], rerefed_raw.info['ch_names'])
        self._remove_conflicting_channels(self.raw, conflicting_channels)

        self.raw.add_channels([rerefed_raw])


    def _add_rereferencing_info(self,
        reref_types: dict
        ) -> None:

        if not self._rereferenced:
            self.extra_info['rereferencing_types'] = {}
        self.extra_info['rereferencing_types'].update(reref_types)

    def _rereference(self,
        RerefMethod: Reref,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        ch_coords_new: list,
        reref_types: list
        ) -> None:

        rerefed_raw, reref_types_dict = self._apply_rereference(
            RerefMethod,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_coords_new,
            reref_types
            )
        self._append_rereferenced_raw(rerefed_raw)
        self._add_rereferencing_info(reref_types_dict)


    def rereference_bipolar(self,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        ch_coords_new: list,
        reref_types: list
        ) -> None:

        self._rereference(
            RerefBipolar,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_coords_new,
            reref_types
        )
        

    def rereference_CAR(self,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        ch_coords_new: list,
        reref_types: list
        ) -> None:

        self._rereference(
            RerefCAR,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_coords_new,
            reref_types
        )
    

    def rereference_pseudo(self,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        ch_coords_new: list,
        reref_types: list
        ) -> None:

        self._rereference(
            RerefPseudo,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_coords_new,
            reref_types
        )


