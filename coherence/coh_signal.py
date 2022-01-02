import mne_bids
import numpy as np
import mne

from coh_rereference import Reref, RerefBipolar, RerefCAR, RerefPseudo




class Signal:

    def __init__(self,
        verbose: bool = True
        ) -> None:

        setattr(self, '_verbose', verbose)
        self._initialise_attributes()
        self._initialise_entries()


    def _initialise_entries(self) -> None:

        self.processing_steps = []
        self.extra_info = {}
        self.extra_info['rereferencing_types'] = {}


    def _initialise_attributes(self) -> None:

        setattr(self, '_data_loaded', False)
        setattr(self, '_annotations_loaded', False)
        setattr(self, '_channels_picked', False)
        setattr(self, '_coordinates_set', False)
        setattr(self, '_bandpass_filtered', False)
        setattr(self, '_notch_filtered', False)
        setattr(self, '_resampled', False)
        setattr(self, '_rereferenced', False)
        setattr(self, '_rereferenced_bipolar', False)
        setattr(self, '_rereferenced_CAR', False)
        setattr(self, '_rereferenced_pseudo', False)
        setattr(self, '_epoched', False)


    def _updateattr(self,
        class_object: object,
        attribute: str,
        value
        ) -> None:

        if hasattr(class_object, attribute):
            setattr(class_object, attribute, value)
        else:
            raise Exception(
                "Error when attempting to update an attribute of "
                f"{class_object}:\nThe attribute {attribute} does not exist, "
                "and so cannot be updated."
            )

    
    def _update_processing_steps(self,
        step_name: str,
        step_value
        ) -> None:

        self.processing_steps.append([step_name, step_value])


    def order_channels(self,
        ch_names: list
        ) -> None:

        self.data.reorder_channels(ch_names)


    def get_coordinates(self) -> list:

        return self.data._get_channel_positions().copy().tolist()


    def _discard_missing_coordinates(self,
        ch_names: list,
        ch_coords: list
        ) -> tuple[list, list]:

        keep_i = [i for i, coords in enumerate(ch_coords) if coords != []]
        return (
            [name for i, name in enumerate(ch_names) if i in keep_i],
            [coords for i, coords in enumerate(ch_coords) if i in keep_i]
        )


    def set_coordinates(self,
        ch_names: list,
        ch_coords: list
        ) -> None:

        ch_names, ch_coords = self._discard_missing_coordinates(
            ch_names, ch_coords
        )
        self.data._set_channel_positions(ch_coords, ch_names)

        self._updateattr(self, '_coordinates_set', True)
        if self._verbose:
            print(f"Setting channel coordinates to:\n{ch_coords}.")


    def get_data(self) -> np.array:

        return self.data.get_data(reject_by_annotation='omit').copy()


    def load_raw(self,
        path_raw: mne_bids.BIDSPath
        ) -> None:

        if self._data_loaded:
            raise Exception(
                "Error when trying to load raw data:\nRaw data has already "
                "been loaded into the object."
            )

        self._path_raw = path_raw
        self.data = mne_bids.read_raw_bids(
            bids_path=self._path_raw, verbose=False
        )
        self.data.load_data()

        self._updateattr(self, '_data_loaded', True)
        self.extra_info['rereferencing_types'].update(
            {ch_name: 'none' for ch_name in self.data.info['ch_names']}
        )
        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.")

    
    def load_annotations(self,
        path_annots: str
        ) -> None:

        if self._epoched:
            raise Exception(
                "Error when adding annotations to the data:\nAnnotations "
                "should be added to the raw data, however the data in this "
                "class has been epoched."
            )

        if self._verbose:
            print(
                "Applying annotations to the data from the filepath:\n"
                f"{path_annots}."
            )

        try:
            self.data.set_annotations(mne.read_annotations(path_annots))
        except:
            print("There are no events to read from the annotations file.")

        self._updateattr(self, '_annotations_loaded', True)
        self._update_processing_steps('annotations_added', True)


    def _pick_extra_info(self,
        ch_names: list
        ) -> None:

        for key in self.extra_info.keys():
            new_entry = {
                ch_name: self.extra_info[key][ch_name] for ch_name in ch_names
            }
            self.extra_info[key] = new_entry


    def _drop_extra_info(self,
        ch_names: list
        ) -> None:

        for key in self.extra_info.keys():
            [self.extra_info[key].pop(name) for name in ch_names]

    
    def _drop_channels(self,
        ch_names: list
        ) -> None:

        self.data.drop_channels(ch_names)
        self._drop_extra_info(ch_names)


    def pick_channels(self,
        ch_names: list
        ) -> None:

        self.data.pick_channels(ch_names)
        self._pick_extra_info(ch_names)

        self._updateattr(self, '_channels_picked', True)
        self._update_processing_steps('channel_picks', ch_names)
        if self._verbose:
            print(
                "Picking specified channels from the data.\nChannels: "
                f"{ch_names}."
            )

    
    def bandpass_filter(self,
        lowpass_freq: int,
        highpass_freq: int
        ) -> None:

        self.data.filter(highpass_freq, lowpass_freq)

        self._updateattr(self, '_bandpass_filtered', True)
        self._update_processing_steps(
            'bandpass_filter', [lowpass_freq, highpass_freq]
        )
        if self._verbose:
            print(
                f"Bandpass filtering the data.\nLow frequency: {highpass_freq} "
                f"Hz. High frequency: {lowpass_freq} Hz."
            )

    
    def notch_filter(self,
        line_noise_freq: int
        ) -> None:

        freqs = np.arange(
            line_noise_freq, self.data.info['lowpass'], line_noise_freq,
            dtype=int
        ).tolist()
        self.data.notch_filter(freqs)

        self._updateattr(self, '_notch_filtered', True)
        self._update_processing_steps('notch_filter', freqs)
        if self._verbose:
            print(
                "Notch filtering the data with line noise frequency "
                f"{line_noise_freq} Hz at the following frequencies (Hz): "
                f"{freqs}."
            )


    def resample(self,
        resample_freq: int
        ) -> None:

        self.data.resample(resample_freq)

        self._updateattr(self, '_resampled', True)
        self._update_processing_steps('resample', resample_freq)
        if self._verbose:
            print(f"Resampling the data at {resample_freq} Hz.")


    def drop_unrereferenced_channels(self) -> None:

        self._drop_channels(
            [ch_name for ch_name in 
            self.extra_info['rereferencing_types'].keys()
            if self.extra_info['rereferencing_types'][ch_name] == 'none']
        )


    def _apply_rereference(self,
        RerefMethod: Reref,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list
        ) -> list:

        RerefObject = RerefMethod(
            self.data.copy(),
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
        ch_names: list
        ) -> None:

        self._drop_channels(ch_names)
        print(
            "Warning when rereferencing data:\nThe following rereferenced "
            f"channels {ch_names} are already present in the raw data.\n"
            "Removing the channels from the raw data."
        )

        
    def _append_rereferenced_raw(self,
        rerefed_raw: mne.io.Raw
        ) -> None:

        ch_names = self._check_conflicting_channels(
            self.data.info['ch_names'], rerefed_raw.info['ch_names']
        )
        if ch_names != []:
            self._remove_conflicting_channels(ch_names)
            
        self.data.add_channels([rerefed_raw])


    def _add_rereferencing_info(self,
        reref_types: dict
        ) -> None:

        self.extra_info['rereferencing_types'].update(reref_types)


    def _get_channel_rereferencing_pairs(self,
        ch_names_old: list,
        ch_names_new: list
        ) -> list:

        return [
            [ch_names_old[i], ch_names_new[i]] for i in range(len(ch_names_old))
        ]


    def _rereference(self,
        RerefMethod: Reref,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list
        ) -> None:

        if self._epoched:
            raise Exception(
                "Error when rereferencing the data:\nThe data to rereference "
                "should be raw, but it has been epoched."
            )

        rerefed_raw, reref_types_dict = self._apply_rereference(
            RerefMethod,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            reref_types,
            ch_coords_new
            )
        self._append_rereferenced_raw(rerefed_raw)
        self._add_rereferencing_info(reref_types_dict)

        self._updateattr(self, '_rereferenced', True)


    def rereference_bipolar(self,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list
        ) -> None:

        self._rereference(
            RerefBipolar,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            reref_types,
            ch_coords_new
        )

        self._updateattr(self, '_rereferenced_bipolar', True)
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps('rereferencing_bipolar', ch_reref_pairs)
        if self._verbose:
            print(f"The following channels have been bipolar rereferenced:")
            [print(f"{old[0]} - {old[1]} -> {new}") for [old, new] in ch_reref_pairs]
        

    def rereference_CAR(self,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list
        ) -> None:

        self._rereference(
            RerefCAR,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            reref_types,
            ch_coords_new,
        )

        self._updateattr(self, '_rereferenced_CAR', True)
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps('rereferencing_CAR', ch_reref_pairs)
        if self._verbose:
            print(f"The following channels have been CAR rereferenced:")
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]
    

    def rereference_pseudo(self,
        ch_names_old: list,
        ch_names_new: list,
        ch_types_new: list,
        reref_types: list,
        ch_coords_new: list
        ) -> None:

        self._rereference(
            RerefPseudo,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            reref_types,
            ch_coords_new
        )

        self._updateattr(self, '_rereferenced_pseudo', True)
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps('rereferencing_pseudo', ch_reref_pairs)
        if self._verbose:
            print(f"The following channels have been pseudo rereferenced:")
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]


    def epoch(self,
        epoch_length: int
        ) -> None:

        if self._epoched:
            raise Exception(
                "Error when epoching data:\nThe data has already been epoched."
            )

        self.data = mne.make_fixed_length_epochs(self.data, epoch_length)

        self._updateattr(self, '_epoched', True)
        self._update_processing_steps('epoch_data', epoch_length)
        if self._verbose:
            print(
                f"Epoching the data with epoch lengths of {epoch_length}"
                "seconds."
            )


