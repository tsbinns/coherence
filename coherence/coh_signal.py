import mne
import mne_bids
import numpy as np
from typing import Any, Option, Union

from coh_dtypes import realnum
from coh_exceptions import ProcessingOrderError, MissingAttributeError
from coh_rereference import Reref, RerefBipolar, RerefCAR, RerefPseudo




class Signal:
    """Class for loading, preprocessing, and epoching an mne.io.Raw object.

    PARAMETERS
    ----------
    verbose : bool | optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    order_channels
    -   Orders channels in the mne.io.Raw or mne.Epochs object based on a
        given order.

    get_coordinates
    -   Extracts coordinates of the channels from the mne.io.Raw or mne.Epochs
        object.

    set_coordinates
    -   Assigns coordinates to the channels in the mne.io.Raw or mne.Epochs
        object.

    get_data
    -   Extracts the data array from the mne.io.Raw or mne.Epochs object,
        excluding data based on the annotations.

    load_raw
    -   Loads an mne.io.Raw object, loads it into memory, and sets it as the
        data, also assigning rereferencing types in 'extra_info' for the
        channels present in the mne.io.Raw object to 'none'.

    load_annotations
    -   Loads annotations corresponding to the mne.io.Raw object.

    pick_channels
    -   Retains only certain channels in the mne.io.Raw or mne.Epochs object,
        also retaining only entries for these channels from the 'extra_info'.

    bandpass_filter
    -   Bandpass filters the mne.io.Raw or mne.Epochs object.

    notch_filter
    -   Notch filters the mne.io.Raw or mne.Epochs object.

    resample
    -   Resamples the mne.io.Raw or mne.Epochs object.

    drop_unrereferenced_channels
    -   Drops channels that have not been rereferenced from the mne.io.Raw or
        mne.Epochs object, also discarding entries for these channels from
        'extra_info'.

    rereference_bipolar
    -   Bipolar rereferences channels in the mne.io.Raw object.

    rereference_CAR
    -   Common-average rereferences channels in the mne.io.Raw object.

    rereference_pseudo
    -   Pseudo rereferences channels in the mne.io.Raw object.
    -   This allows e.g. rereferencing types, channel coordinates, etc... to be
        assigned to the channels without any rereferencing occuring.
    -   This is useful if e.g. the channels were already hardware rereferenced.

    epoch
    -   Divides the mne.io.Raw object into epochs of a specified duration.
    """

    def __init__(self,
        verbose: bool = True
        ) -> None:

        setattr(self, '_verbose', verbose)
        self._initialise_attributes()
        self._initialise_entries()


    def _initialise_entries(self) -> None:
        """Initialises aspects of the Signal object that will be filled with
        information as the data is processed.   
        """

        self.processing_steps = []
        self.extra_info = {}
        self.extra_info['rereferencing_types'] = {}


    def _initialise_attributes(self) -> None:
        """Initialises aspects of the Signal object that indicate which methods
        have been called (starting as 'False'), which can later be updated.
        """

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
        attribute: str,
        value: Any
        ) -> None:
        """Updates aspects of the Signal object that indicate which methods
        have been called.
        -   The aspects must have already been instantiated.

        PARAMETERS
        ----------
        attribute : str
        -   The name of the aspect to update.

        value : Any
        -   The value to update the attribute with.

        RAISES
        ------
        MissingAttributeError
        -   Raised if the user attempts to update an attribute that has not been
            instantiated in '_instantiate_attributes'.
        """

        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise MissingAttributeError(
                f"Error when attempting to update an attribute of {self}:\nThe "
                f"attribute {attribute} does not exist, and so cannot be "
                "updated."
            )

    
    def _update_processing_steps(self,
        step_name: str,
        step_value: Any
        ) -> None:
        """Updates the 'processing_steps' aspect of the Signal object with new
        information.

        PARAMETERS
        ----------
        step_name : str
        -   The name of the processing step.

        step_value : Any
        -   A value representing what processing has taken place.
        """

        self.processing_steps.append([step_name, step_value])


    def order_channels(self,
        ch_names: list[str]
        ) -> None:
        """Orders channels in the mne.io.Raw or mne.Epochs object based on a
        given order.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   A list of channel names in the mne.io.Raw or mne.Epochs object in
            the order that you want the channels to be ordered.
        """

        self.data.reorder_channels(ch_names)


    def get_coordinates(self) -> list[list[realnum]]:
        """Extracts coordinates of the channels from the mne.io.Raw or
        mne.Epochs object.

        RETURNS
        -------
        list[list[int or float]]
        -   List of the channel coordinates, with each list entry containing the
            x, y, and z coordinates of each channel.
        """

        return self.data._get_channel_positions().copy().tolist()


    def _discard_missing_coordinates(self,
        ch_names: list[str],
        ch_coords: list[list[realnum]]
        ) -> tuple[list, list]:
        """Removes empty sublists from a parent list of channel coordinates
        (also removes them from the corresponding entries of channel names)
        before applying the coordinates to the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels corresponding to the coordinates in
            'ch_coords'.

        ch_coords : list[empty list or list[int or float]]
        -   Coordinates of the channels, with each entry consiting of a sublist
            containing the x, y, and z coordinates of the corresponding channel
            specified in 'ch_names', or being empty.

        RETURNS
        -------
        empty list or list[str]
        -   Names of the channels corresponding to the coordinates in
            'ch_coords', with those names corresponding to empty sublists (i.e
            missing coordinates) in 'ch_coords' having been removed.
        
        empty list or list[list[int or float]]
        -   Coordinates of the channels corresponding the the channel names in
            'ch_names', with the empty sublists (i.e missing coordinates) having
            been removed. 
        """

        keep_i = [i for i, coords in enumerate(ch_coords) if coords != []]
        return (
            [name for i, name in enumerate(ch_names) if i in keep_i],
            [coords for i, coords in enumerate(ch_coords) if i in keep_i]
        )


    def set_coordinates(self,
        ch_names: list[str],
        ch_coords: list[list[realnum]]
        ) -> None:
        """Assigns coordinates to the channels in the mne.io.Raw or mne.Epochs
        object.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels corresponding to the coordinates in
            'ch_coords'.

        ch_coords : list[empty list or list[int or float]]
        -   Coordinates of the channels, with each entry consiting of a sublist
            containing the x, y, and z coordinates of the corresponding channel
            specified in 'ch_names'.
        """

        ch_names, ch_coords = self._discard_missing_coordinates(
            ch_names, ch_coords
        )
        self.data._set_channel_positions(ch_coords, ch_names)

        self._updateattr('_coordinates_set', True)
        if self._verbose:
            print(f"Setting channel coordinates to:\n{ch_coords}.")


    def get_data(self) -> np.array:
        """Extracts the data array from the mne.io.Raw or mne.Epochs object,
        excluding data based on the annotations.

        RETURNS
        -------
        np.array
        -   Array of the data.
        """

        return self.data.get_data(reject_by_annotation='omit').copy()


    def load_raw(self,
        path_raw: mne_bids.BIDSPath
        ) -> None:
        """Loads an mne.io.Raw object, loads it into memory, and sets it as the
        data, also assigning rereferencing types in 'extra_info' for the
        channels present in the mne.io.Raw object to 'none'.

        PARAMETERS
        ----------
        path_raw : mne_bids.BIDSPath
        -   The path of the raw data to be loaded.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to load an mne.io.Raw object into the
            self object if an mne.io.Raw object has already been loaded.
        -   A new Signal object should be instantiated and used instead.
        """

        if self._data_loaded:
            raise ProcessingOrderError(
                "Error when trying to load raw data:\nRaw data has already "
                "been loaded into the object."
            )

        self._path_raw = path_raw
        self.data = mne_bids.read_raw_bids(
            bids_path=self._path_raw, verbose=False
        )
        self.data.load_data()

        self._updateattr('_data_loaded', True)
        self.extra_info['rereferencing_types'].update(
            {ch_name: 'none' for ch_name in self.data.info['ch_names']}
        )
        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.")

    
    def load_annotations(self,
        path_annots: str
        ) -> None:
        """Loads annotations corresponding to the mne.io.Raw object.

        PARAMETERS
        ----------
        path_annots : str
        -   The filepath of the annotations to load.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to load annotations into the data after
            it has been epoched.
        -   Annotations should be loaded before epoching has occured, when the 
            data is in the form of an mne.io.Raw object rather than an 
            mne.Epochs object.
        """

        if self._epoched:
            raise ProcessingOrderError(
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

        self._updateattr('_annotations_loaded', True)
        self._update_processing_steps('annotations_added', True)


    def _pick_extra_info(self,
        ch_names: list[str]
        ) -> None:
        """Retains entries for selected channels in 'extra_info', discarding
        those for the remaining channels.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels whose entries should be retained.
        """

        for key in self.extra_info.keys():
            new_entry = {
                ch_name: self.extra_info[key][ch_name] for ch_name in ch_names
            }
            self.extra_info[key] = new_entry


    def _drop_extra_info(self,
        ch_names: list[str]
        ) -> None:
        """Removes entries for selected channels in 'extra_info', retaining
        those for the remaining channels.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels whose entries should be discarded.
        """

        for key in self.extra_info.keys():
            [self.extra_info[key].pop(name) for name in ch_names]

    
    def _drop_channels(self,
        ch_names: list[str]
        ) -> None:
        """Removes channels from the mne.io.Raw or mne.Epochs object, as well as
        from entries in 'extra_info'.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that should be discarded.
        """

        self.data.drop_channels(ch_names)
        self._drop_extra_info(ch_names)


    def pick_channels(self,
        ch_names: list[str]
        ) -> None:
        """Retains only certain channels in the mne.io.Raw or mne.Epochs object,
        also retaining only entries for these channels from the 'extra_info'.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that should be retained.
        """

        self.data.pick_channels(ch_names)
        self._pick_extra_info(ch_names)

        self._updateattr('_channels_picked', True)
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
        """Bandpass filters the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        lowpass_freq : int
        -   The frequency (Hz) at which to lowpass filter the data.

        highpass_freq : int
        -   The frequency (Hz) at which to highpass filter the data.
        """

        self.data.filter(highpass_freq, lowpass_freq)

        self._updateattr('_bandpass_filtered', True)
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
        """Notch filters the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        line_noise : int
        -   The line noise frequency (Hz) for which the notch filter, including
            the harmonics, is produced.
        """

        freqs = np.arange(
            line_noise_freq, self.data.info['lowpass'], line_noise_freq,
            dtype=int
        ).tolist()
        self.data.notch_filter(freqs)

        self._updateattr('_notch_filtered', True)
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
        """Resamples the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        resample_freq : int
        -   The frequency at which to resample the data.
        """

        self.data.resample(resample_freq)

        self._updateattr('_resampled', True)
        self._update_processing_steps('resample', resample_freq)
        if self._verbose:
            print(f"Resampling the data at {resample_freq} Hz.")


    def drop_unrereferenced_channels(self) -> None:
        """Drops channels that have not been rereferenced from the mne.io.Raw or
        mne.Epochs object, also discarding entries for these channels from
        'extra_info'.
        """

        self._drop_channels(
            [ch_name for ch_name in 
            self.extra_info['rereferencing_types'].keys()
            if self.extra_info['rereferencing_types'][ch_name] == 'none']
        )


    def _apply_rereference(self,
        RerefMethod: Reref,
        ch_names_old: Union[list[str], list[list[str]]],
        ch_names_new: Option[list[Option[str]]],
        ch_types_new: Option[list[Option[str]]],
        reref_types: Option[list[Option[str]]],
        ch_coords_new: Option[list[Option[list[realnum]]]]
        ) -> tuple[mne.io.Raw, dict[str, str]]:
        """Applies a rereferencing method to the mne.io.Raw object.

        PARAMETERS
        ----------
        RerefMethod : Reref
        -   The rereferencing method to apply.

        ch_names_old : list[str or list[str]]
        -   The names of the channels in the mne.io.Raw object to rereference.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

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

        RETURNS
        -------
        mne.io.Raw
        -   The rereferenced data in an mne.io.Raw object.

        dict[str, str]
        -   Dictionary showing the rereferencing types applied to the channels, 
            in which the key:value pairs are channel name : rereference type.
        """

        RerefObject = RerefMethod(
            self.data.copy(), ch_names_old, ch_names_new, ch_types_new,
            reref_types, ch_coords_new
        )

        return RerefObject.rereference()


    def _check_conflicting_channels(self,
        ch_names_1: list[str],
        ch_names_2: list[str]
        ) -> list:
        """Checks whether there are any of the same channel names in two lists.
        -   Useful to perform before appending an external mne.io.Raw or
            mne.Epochs object.

        PARAMETERS
        ----------
        ch_names_1 : list[str]
        -   List of channel names to compare.

        ch_names_2 : list[str]
        -   Another list of channel names to compare.

        RETURNS
        -------
        empty list or list[str]
        -   Names that are present in both channels.
        """

        return [name for name in ch_names_1 if name in ch_names_2]


    def _remove_conflicting_channels(self,
        ch_names: list[str]
        ) -> None:
        """Removes channels from the self mne.io.Raw or mne.Epochs object. 
        -   Designed for use alongside '_append_rereferenced_raw'.
        -   Useful to perform before appending an external mne.io.Raw or
            mne.Epochs object.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of channels to remove from the mne.io.Raw or mne.Epochs
            object.
        """

        self._drop_channels(ch_names)
        print(
            "Warning when rereferencing data:\nThe following rereferenced "
            f"channels {ch_names} are already present in the raw data.\n"
            "Removing the channels from the raw data."
        )

        
    def _append_rereferenced_raw(self,
        rerefed_raw: mne.io.Raw
        ) -> None:
        """Appends a rereferenced mne.io.Raw object to the self mne.io.Raw
        object, first discarding channels in the self mne.io.Raw object which
        have the same names as those in the mne.io.Raw object to append.
        
        PARAMETERS
        ----------
        rerefed_raw : mne.io.Raw
        -   An mne.io.Raw object that has been rereferenced which will be
            appended.
        """

        ch_names = self._check_conflicting_channels(
            self.data.info['ch_names'], rerefed_raw.info['ch_names']
        )
        if ch_names != []:
            self._remove_conflicting_channels(ch_names)
            
        self.data.add_channels([rerefed_raw])


    def _add_rereferencing_info(self,
        reref_types: dict[str, str]
        ) -> None:
        """Adds channel rereferencing information to 'extra_info'.

        PARAETERS
        ---------
        reref_types : dict[str, str]
        -   Dictionary with key:value pairs of channel name : rereferencing
            type.
        """

        self.extra_info['rereferencing_types'].update(reref_types)


    def _get_channel_rereferencing_pairs(self,
        ch_names_old: Union[list[str], list[list[str]]],
        ch_names_new: list[str]
        ) -> list:
        """Collects the names of the channels that were referenced and the newly
        generated channels together.

        PARAMETERS
        ----------
        ch_names_old : list[str or list[str]]
        -   Names of the channels that were rereferenced.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

        ch_names_new : list[str]
        -   Names of the channels that were produced by the rereferencing.
        
        RETURNS
        -------
        list
        -   List of sublists, in which each sublist contains the name(s) of the
            channel(s) that was(were) rereferenced, and the name of the channel
            that was produced.
        """

        return [
            [ch_names_old[i], ch_names_new[i]] for i in range(len(ch_names_old))
        ]


    def _rereference(self,
        RerefMethod: Reref,
        ch_names_old: Union[list[str], list[list[str]]],
        ch_names_new: Option[list[Option[str]]],
        ch_types_new: Option[list[Option[str]]],
        reref_types: Option[list[Option[str]]],
        ch_coords_new: Option[list[Option[list[realnum]]]]
        ) -> None:
        """Parent method for calling on other methods to rereference the data,
        add it to the self mne.io.Raw object, and add the rereferecing
        information to 'extra_info'.

        PARAMETERS
        ----------
        RerefMethod : Reref
        -   The rereferencing method to apply.

        ch_names_old : list[str or list[str]]
        -   The names of the channels in the mne.io.Raw object to rereference.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

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

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to rereference the data after it has
            already been epoched.
        -   Rereferencing should only be applied when the data in self is held
            as an mne.io.Raw object rather than an mne.Epochs object.
        """

        if self._epoched:
            raise ProcessingOrderError(
                "Error when rereferencing the data:\nThe data to rereference "
                "should be raw, but it has been epoched."
            )

        rerefed_raw, reref_types_dict = self._apply_rereference(
            RerefMethod, ch_names_old, ch_names_new, ch_types_new, reref_types,
            ch_coords_new
        )
        self._append_rereferenced_raw(rerefed_raw)
        self._add_rereferencing_info(reref_types_dict)

        self._updateattr('_rereferenced', True)


    def rereference_bipolar(self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: list[str],
        reref_types: list[str],
        ch_coords_new: Option[list[Option[list[realnum]]]]
        ) -> None:
        """Bipolar rereferences channels in the mne.io.Raw object.

        PARAMETERS
        ----------
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
        """

        self._rereference(
            RerefBipolar, ch_names_old, ch_names_new, ch_types_new, reref_types,
            ch_coords_new
        )

        self._updateattr('_rereferenced_bipolar', True)
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps('rereferencing_bipolar', ch_reref_pairs)
        if self._verbose:
            print(f"The following channels have been bipolar rereferenced:")
            [print(f"{old[0]} - {old[1]} -> {new}") for [old, new] in ch_reref_pairs]
        

    def rereference_CAR(self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: list[str],
        reref_types: list[str],
        ch_coords_new: Option[list[Option[list[realnum]]]]
        ) -> None:
        """Common-average rereferences channels in the mne.io.Raw object.

        PARAMETERS
        ----------
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
        """

        self._rereference(
            RerefCAR, ch_names_old, ch_names_new, ch_types_new, reref_types,
            ch_coords_new,
        )

        self._updateattr('_rereferenced_CAR', True)
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps('rereferencing_CAR', ch_reref_pairs)
        if self._verbose:
            print(f"The following channels have been CAR rereferenced:")
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]
    

    def rereference_pseudo(self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: list[str],
        reref_types: list[str],
        ch_coords_new: Option[list[Option[list[realnum]]]]
        ) -> None:
        """Pseudo rereferences channels in the mne.io.Raw object.
        -   This allows e.g. rereferencing types, channel coordinates, etc... to
            be assigned to the channels without any rereferencing occuring.
        -   This is useful if e.g. the channels were already hardware
            rereferenced.

        PARAMETERS
        ----------
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
        """

        self._rereference(
            RerefPseudo, ch_names_old, ch_names_new, ch_types_new, reref_types,
            ch_coords_new
        )

        self._updateattr('_rereferenced_pseudo', True)
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
        """Divides the mne.io.Raw object into epochs of a specified duration.

        PARAMETERS
        ----------
        epoch_length : int
        -   The duration of the epochs (seconds) to divide the data into.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to epoch the data once it has already
            been epoched.
        -   This method can only be called if the data is stored as an 
            mne.io.Raw object, not as an mne.Epochs object.
        """

        if self._epoched:
            raise ProcessingOrderError(
                "Error when epoching data:\nThe data has already been epoched."
            )

        self.data = mne.make_fixed_length_epochs(self.data, epoch_length)
        self.data.load_data()

        self._updateattr('_epoched', True)
        self._update_processing_steps('epoch_data', epoch_length)
        if self._verbose:
            print(
                f"Epoching the data with epoch lengths of {epoch_length}"
                "seconds."
            )


