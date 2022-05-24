"""A class for loading, preprocessing, and epoching an mne.io.Raw object.

CLASSES
-------
Signal
-   Class for loading, preprocessing, and epoching an mne.io.Raw object.
"""


from copy import deepcopy
from typing import Any, Optional, Union
from numpy.typing import NDArray
from mne import concatenate_epochs
import mne
import mne_bids
import numpy as np
from coh_exceptions import (
    ChannelAttributeError,
    EntryLengthError,
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_rereference import Reref, RerefBipolar, RerefCommonAverage, RerefPseudo
from coh_handle_entries import (
    check_lengths_list_identical,
    check_repeated_vals,
    ordered_dict_keys_from_list,
    ordered_list_from_dict,
    rearrange_axes,
)
from coh_handle_files import (
    check_annots_empty,
    check_ftype_present,
    identify_ftype,
)
from coh_handle_objects import create_extra_info, create_mne_data_object
from coh_saving import check_before_overwrite, save_as_json, save_as_pkl


class Signal:
    """Class for loading, preprocessing, and epoching an mne.io.Raw object.

    PARAMETERS
    ----------
    verbose : bool; default True
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

    combine_channels
    -   Combines the data of multiple channels in the mne.io.Raw object through
        addition and adds this combined data as a new channel.

    drop_unrereferenced_channels
    -   Drops channels that have not been rereferenced from the mne.io.Raw or
        mne.Epochs object, also discarding entries for these channels from
        'extra_info'.

    rereference_bipolar
    -   Bipolar rereferences channels in the mne.io.Raw object.

    rereference_common_average
    -   Common-average rereferences channels in the mne.io.Raw object.

    rereference_pseudo
    -   Pseudo rereferences channels in the mne.io.Raw object.
    -   This allows e.g. rereferencing types, channel coordinates, etc... to be
        assigned to the channels without any rereferencing occuring.
    -   This is useful if e.g. the channels were already hardware rereferenced.

    epoch
    -   Divides the mne.io.Raw object into epochs of a specified duration.

    save_object
    -   Saves the Signal object as a .pkl file.

    save_signals
    -   Saves the time-series data and additional information as a file.
    """

    def __init__(self, verbose: bool = True) -> None:

        # Initialises aspects of the Signal object that will be filled with
        # information as the data is processed.
        self.processing_steps = {"preprocessing": {}}
        self._processing_step_number = 1
        self.extra_info = {}
        self.data = [None]
        self._path_raw = None
        self.data_dimensions = None

        # Initialises inputs of the Signal object.
        self._verbose = verbose

        # Initialises aspects of the Signal object that indicate which methods
        # have been called (starting as 'False'), which can later be updated.
        self._data_loaded = False
        self._annotations_loaded = False
        self._channels_picked = False
        self._coordinates_set = False
        self._regions_set = False
        self._hemispheres_set = False
        self._bandpass_filtered = False
        self._notch_filtered = False
        self._resampled = False
        self._rereferenced = False
        self._rereferenced_bipolar = False
        self._rereferenced_common_average = False
        self._rereferenced_pseudo = False
        self._windowed = False
        self._epoched = False
        self._shuffled = False

    def _update_processing_steps(self, step_name: str, step_value: Any) -> None:
        """Updates the 'preprocessing' entry of the 'processing_steps'
        dictionary of the Signal object with new information consisting of a
        key:value pair in which the key is numbered based on the applied steps.

        PARAMETERS
        ----------
        step_name : str
        -   The name of the processing step.

        step_value : Any
        -   A value representing what processing has taken place.
        """

        step_name = f"{self._processing_step_number}.{step_name}"
        self.processing_steps["preprocessing"][step_name] = step_value
        self._processing_step_number += 1

    def add_metadata(self, metadata: dict) -> None:
        """Adds information about the data being preprocessed to the extra_info
        aspect.

        PARAMETERS
        ----------
        metadata : dict
        -   Information about the data being preprocessed.
        """

        self.extra_info["metadata"] = metadata

    def _order_extra_info(self, order: list[str]) -> None:
        """Order channels in 'extra_info'.

        PARAMETERS
        ----------
        order : list[str]
        -   The order in which the channels should appear in the attributes of
            the 'extra_info' dictionary.
        """

        to_order = [
            "ch_reref_types",
            "ch_regions",
            "ch_hemispheres",
            "ch_epoch_orders",
        ]
        for key in to_order:
            if key in self.extra_info.keys():
                self.extra_info[key] = ordered_dict_keys_from_list(
                    dict_to_order=self.extra_info[key], keys_order=order
                )

    def order_channels(self, ch_names: list[str]) -> None:
        """Orders channels in the mne.io.Raw or mne.Epochs objects, as well as
        the 'extra_info' dictionary, based on a given order.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   A list of channel names in the mne.io.Raw or mne.Epochs object in
            the order that you want the channels to be ordered.
        """

        for data in self.data:
            data.reorder_channels(ch_names)
        self._order_extra_info(order=ch_names)

        if self._verbose:
            print("Reordering the channels in the following order:")
            [print(name) for name in ch_names]

    def get_coordinates(
        self, picks: Union[str, list, slice, None] = None
    ) -> list[list[Union[int, float]]]:
        """Extracts coordinates of the channels from the mne.io.Raw or
        mne.Epochs objects.

        PARAMETERS
        ----------
        picks : str | list | slice | None; default None
        -   Selects which channels' coordinates should be returned.
        -   If 'None', returns coordinates for all good channels.

        RETURNS
        -------
        list[list[int or float]]
        -   List of the channel coordinates, with each list entry containing the
            x, y, and z coordinates of each channel.
        """

        return self.data[0]._get_channel_positions(picks=picks).copy().tolist()

    def _discard_missing_coordinates(
        self, ch_names: list[str], ch_coords: list[list[Union[int, float]]]
    ) -> tuple[list, list]:
        """Removes empty sublists from a parent list of channel coordinates
        (also removes them from the corresponding entries of channel names)
        before applying the coordinates to the mne.io.Raw or mne.Epochs objects.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels corresponding to the coordinates in
            'ch_coords'.

        ch_coords : list[empty list | list[int | float]]
        -   Coordinates of the channels, with each entry consiting of a sublist
            containing the x, y, and z coordinates of the corresponding channel
            specified in 'ch_names', or being empty.

        RETURNS
        -------
        empty list | list[str]
        -   Names of the channels corresponding to the coordinates in
            'ch_coords', with those names corresponding to empty sublists (i.e
            missing coordinates) in 'ch_coords' having been removed.

        empty list  |list[list[int | float]]
        -   Coordinates of the channels corresponding the the channel names in
            'ch_names', with the empty sublists (i.e missing coordinates) having
            been removed.
        """

        keep_i = [i for i, coords in enumerate(ch_coords) if coords != []]
        return (
            [name for i, name in enumerate(ch_names) if i in keep_i],
            [coords for i, coords in enumerate(ch_coords) if i in keep_i],
        )

    def set_coordinates(
        self, ch_names: list[str], ch_coords: list[list[Union[int, float]]]
    ) -> None:
        """Assigns coordinates to the channels in the mne.io.Raw or mne.Epochs
        object.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels corresponding to the coordinates in
            'ch_coords'.

        ch_coords : list[empty list | list[int | float]]
        -   Coordinates of the channels, with each entry consiting of a sublist
            containing the x, y, and z coordinates of the corresponding channel
            specified in 'ch_names'.
        """

        ch_names, ch_coords = self._discard_missing_coordinates(
            ch_names, ch_coords
        )
        for data in self.data:
            data._set_channel_positions(ch_coords, ch_names)

        self._coordinates_set = True
        if self._verbose:
            print("Setting channel coordinates to:")
            [
                print(f"{ch_names[i]}: {ch_coords[i]}")
                for i in range(len(ch_names))
            ]

    def set_regions(self, ch_names: list[str], ch_regions: list[str]) -> None:
        """Adds channel regions to the extra_info dictionary.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels corresponding to the regions in
            'ch_regions'.

        ch_regions : list[str]
        -   Regions of the channels, with each entry consiting of a region name
            of the corresponding channel specified in 'ch_names'.
        """

        if len(ch_names) != len(ch_regions):
            raise EntryLengthError(
                "The channel names and regions do not have the same length "
                f"({len(ch_names)} and {len(ch_regions)}, respectively)."
            )

        for i, ch_name in enumerate(ch_names):
            self.extra_info["ch_regions"][ch_name] = ch_regions[i]
        for i in range(len(self.data)):
            self.data[i].ch_regions = self.extra_info["ch_regions"]

        self._regions_set = True
        if self._verbose:
            print("Setting channel regions to:")
            [
                print(f"{ch_names[i]}: {ch_regions[i]}")
                for i in range(len(ch_names))
            ]

    def set_hemispheres(
        self, ch_names: list[str], ch_hemispheres: list[str]
    ) -> None:
        """Adds channel hemispheres to the extra_info dictionary.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels.

        ch_hemispheres : list[str]
        -   Hemispheres of the channels.
        """

        if len(ch_names) != len(ch_hemispheres):
            raise EntryLengthError(
                "The channel names and regions do not have the same length "
                f"({len(ch_names)} and {len(ch_hemispheres)}, respectively)."
            )

        for i, ch_name in enumerate(ch_names):
            self.extra_info["ch_hemispheres"][ch_name] = ch_hemispheres[i]
        for i in range(len(self.data)):
            self.data[i].ch_hemispheres = self.extra_info["ch_hemispheres"]

        self._hemispheres_set = True
        if self._verbose:
            print("Setting channel hemispheres to:")
            [
                print(f"{ch_names[i]}: {ch_hemispheres[i]}")
                for i in range(len(ch_names))
            ]

    def get_data(self) -> np.array:
        """Extracts the data array from the mne.io.Raw or mne.Epochs objects,
        excluding data based on the annotations if the data is an MNE Raw
        object.

        RETURNS
        -------
        data_arr : np.array
        -   Array of the data.
        """

        data_arr = np.empty(len(self.data))
        for i, data in enumerate(self.data):
            if isinstance(data, mne.io.Raw):
                data_arr[i] = data.get_data(reject_by_annotation="omit").copy()
            else:
                data_arr[i] = data.get_data().copy()

        return data_arr

    def _initialise_additional_info(self) -> None:
        """Fills the extra_info dictionary with placeholder information. This
        should only be called when the data is initially loaded.
        """

        info_to_set = ["ch_reref_types", "ch_regions", "ch_hemispheres"]
        for info in info_to_set:
            self.extra_info[info] = {
                ch_name: None for ch_name in self.data[0].info["ch_names"]
            }
        self.data_dimensions = ["windows", "channels", "timepoints"]

    def _fix_coords(self) -> None:
        """Fixes the units of the channel coordinates in the data by multiplying
        them by 1,000."""

        ch_coords = self.get_coordinates()
        for ch_i, coords in enumerate(ch_coords):
            ch_coords[ch_i] = [coord * 1000 for coord in coords]
        self.set_coordinates(
            ch_names=self.data[0].ch_names, ch_coords=ch_coords
        )

    def raw_from_fpath(self, path_raw: mne_bids.BIDSPath) -> None:
        """Loads an mne.io.Raw object from a filepath, loads it into memory, and
        sets it as the data, also assigning additional information in
        'extra_info' for the channels present in the mne.io.Raw object to
        'none'.

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
        self.data[0] = mne_bids.read_raw_bids(
            bids_path=self._path_raw, verbose=False
        )
        self.data[0].load_data()
        self._initialise_additional_info()
        # self._fix_coords()

        self._data_loaded = True
        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.\n")

    def data_from_objects(
        self,
        data: list[Union[mne.io.Raw, mne.Epochs]],
        data_dimensions: list[str],
        processing_steps: dict,
        extra_info: dict,
    ) -> None:
        """Sets the data, its dimensions, processing steps, and additional
        information from their respective objects.

        PARAMETERS
        ----------
        data : list[MNE Raw | MNE Epochs]
        -   Windowed data in an MNE object stored in a list..

        data_dimensions : list[str]
        -   Names of the dimensions in the data. Each window must have the same
            dimensions.

        processing_steps : dict
        -   Information about the processing that has been applied to the data.

        extra_info : dict
        -   Additional information about the data not included in the MNE
            objects.
        """

        if self._data_loaded:
            raise ProcessingOrderError(
                "Error when trying to load data:\nData has already been loaded "
                "into the object."
            )

        self.data = data
        self.data_dimensions = data_dimensions
        self.processing_steps = processing_steps
        self.extra_info = extra_info

        self._set_method_bools()

        self._data_loaded = True
        if self._verbose:
            if isinstance(data[0], mne.io.Raw):
                data_type = "raw"
            else:
                data_type = "epoched"
            print(f"Loading the {data_type} data from a set of objects.\n")

    def _set_method_bools(self) -> None:
        """Sets the markers for methods that have been called on the data based
        on the processing steps and extra information dictionaries."""

        for steps in self.processing_steps.values():
            for step in steps:
                if "rereferencing_common_average" in step:
                    self._rereferenced_common_average = True
                    self._rereferenced = True
                elif "rereferencing_bipolar" in step:
                    self._rereferenced_bipolar = True
                    self._rereferenced = True
                elif "rereferencing_pseudo" in step:
                    self._rereferenced_pseudo = True
                    self._rereferenced = True
                elif "annotations_loaded" in step:
                    self._annotations_loaded = True
                elif "bandpass_filter" in step:
                    self._bandpass_filtered = True
                elif "notch_filter" in step:
                    self._notch_filtered = True
                elif "resample" in step:
                    self._resampled = True
                elif "windowed" in step:
                    self._windowed = True
                elif "epoch_data" in step:
                    self._epoched = True
                elif "shuffle_data" in step:
                    self._shuffled = True

        if self.extra_info["ch_regions"] is not None:
            self._regions_set = True
        if self.extra_info["ch_hemispheres"] is not None:
            self._hemispheres_set = True

    def _remove_bad_annotations(
        self,
    ) -> tuple[mne.annotations.Annotations, int]:
        """Removes segments annotated as 'bad' from the Annotations object.

        RETURNS
        -------
        MNE annotations Annotations
        -   The annotations with 'bad' segments removed.

        int
        -   The number of segments annotated as 'bad'.
        """

        annotations = deepcopy(self.data[0].annotations)
        bad_annot_idcs = []
        for annot_i, annot_name in enumerate(annotations.description):
            if annot_name[:3] == "BAD":
                bad_annot_idcs.append(annot_i)

        return annotations.delete(bad_annot_idcs), len(bad_annot_idcs)

    def _remove_bad_segments(self) -> None:
        """Removes segments annotated as 'bad' from the Raw object."""

        new_annotations, n_bad_segments = self._remove_bad_annotations()
        new_data = self.data[0].get_data(reject_by_annotation="omit")

        self.data[0] = mne.io.RawArray(data=new_data, info=self.data[0].info)
        self.data[0].set_annotations(annotations=new_annotations)

        if self._verbose:
            print(
                f"Removing {n_bad_segments} segment(s) marked as 'bad' from "
                "the data."
            )

    def load_annotations(self, path_annots: str) -> None:
        """Loads annotations corresponding to the mne.io.Raw object.

        PARAMETERS
        ----------
        path_annots : str
        -   The filepath of the annotations to load.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to load annotations into the data after
            it has been windowed.
        -   Raised if the user attempts to load annotations into the data after
            it has been epoched.
        -   Annotations should be loaded before epoching has occured, when the
            data is in the form of an mne.io.Raw object rather than an
            mne.Epochs object.
        """

        if self._windowed:
            raise ProcessingOrderError(
                "Error when adding annotations to the data:\nAnnotations "
                "should be added to the raw data, however the data in this "
                "class has been windowed."
            )
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

        if check_annotations_empty(path_annots):
            print("There are no events to read from the annotations file.")
            annotations_present = False
        else:
            self.data[0].set_annotations(mne.read_annotations(path_annots))
            annotations_present = True

        if annotations_present:
            self._remove_bad_segments()

        self._annotations_loaded = True
        self._update_processing_steps(
            step_name="annotations_loaded", step_value=True
        )

    def _pick_extra_info(self, ch_names: list[str]) -> None:
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

    def _drop_extra_info(self, ch_names: list[str]) -> None:
        """Removes entries for selected channels in 'extra_info', retaining
        those for the remaining channels.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels whose entries should be discarded.
        """

        for key in self.extra_info.keys():
            [self.extra_info[key].pop(name) for name in ch_names]

    def _drop_channels(self, ch_names: list[str]) -> None:
        """Removes channels from the mne.io.Raw or mne.Epochs objects, as well
        as from entries in 'extra_info'.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that should be discarded.
        """

        self._drop_extra_info(ch_names)
        for i, data in enumerate(self.data):
            self.data[i] = data.drop_channels(ch_names)
            self.data[i].ch_regions = self.extra_info["ch_regions"]
            self.data[i].ch_hemispheres = self.extra_info["ch_hemispheres"]

    def pick_channels(self, ch_names: list[str]) -> None:
        """Retains only certain channels in the mne.io.Raw or mne.Epochs
        objects, also retaining only entries for these channels from the
        'extra_info'.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that should be retained.
        """

        self._pick_extra_info(ch_names)
        for i, data in enumerate(self.data):
            data.pick_channels(ch_names)
            self.data[i].ch_regions = self.extra_info["ch_regions"]
            self.data[i].ch_hemispheres = self.extra_info["ch_hemispheres"]

        self._channels_picked = True
        self._update_processing_steps("channel_picks", ch_names)
        if self._verbose:
            print(
                "Picking specified channels from the data.\nChannels: "
                f"{ch_names}."
            )

    def bandpass_filter(self, lowpass_freq: int, highpass_freq: int) -> None:
        """Bandpass filters the mne.io.Raw or mne.Epochs objects.

        PARAMETERS
        ----------
        lowpass_freq : int
        -   The frequency (Hz) at which to lowpass filter the data.

        highpass_freq : int
        -   The frequency (Hz) at which to highpass filter the data.
        """

        for data in self.data:
            data.filter(highpass_freq, lowpass_freq)

        self._bandpass_filtered = True
        self._update_processing_steps(
            "bandpass_filter", [lowpass_freq, highpass_freq]
        )
        if self._verbose:
            print(
                f"Bandpass filtering the data.\nLow frequency: {highpass_freq} "
                f"Hz. High frequency: {lowpass_freq} Hz."
            )

    def notch_filter(self, line_noise_freq: int) -> None:
        """Notch filters the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        line_noise : int
        -   The line noise frequency (Hz) for which the notch filter, including
            the harmonics, is produced.
        """

        freqs = np.arange(
            line_noise_freq,
            self.data[0].info["lowpass"],
            line_noise_freq,
            dtype=int,
        ).tolist()
        for data in self.data:
            data.notch_filter(freqs)

        self._notch_filtered = True
        self._update_processing_steps("notch_filter", freqs)
        if self._verbose:
            print(
                "Notch filtering the data with line noise frequency "
                f"{line_noise_freq} Hz at the following frequencies (Hz): "
                f"{freqs}."
            )

    def resample(self, resample_freq: int) -> None:
        """Resamples the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        resample_freq : int
        -   The frequency, in Hz, at which to resample the data.
        """

        for data in self.data:
            data.resample(resample_freq)

        self._resampled = True
        self._update_processing_steps("resample", resample_freq)
        if self._verbose:
            print(f"Resampling the data at {resample_freq} Hz.")

    def _check_combination_input_lengths(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> None:
        """Checks that the input for combining channels are all of the same
        length.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        ch_names_new : list[str]
        -   The names of the combined channels.

        ch_types_new : list[str | None] | None
        -   The types of the new channels.
        -   If None or if some entries are None, the type is determined based on
            the channels being combined, in which case they must be of the same
            type.

        ch_coords_new : list[list[int | float] | None] | None
        -   The coordinates of the combined channels.
        -   If None or if some entries are None, the coordinates are determined
            based on the channels being combined.

        ch_regions_new : list[str | None] | None
        -   The regions of the new channels.
        -   If None or if some entries are None, the region is determined based
            on the channels being combined, in which case they must be from the
            same region.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the new channels.
        -   If None or if some entries are None, the hemisphere is determined
            based on the channels being combined, in which case they must be
            from the same hemisphere.
        """

        identical, lengths = check_lengths_list_identical(
            to_check=[
                ch_names_old,
                ch_names_new,
                ch_types_new,
                ch_coords_new,
                ch_regions_new,
                ch_hemispheres_new,
            ],
            ignore_values=[None],
        )
        if not identical:
            raise EntryLengthError(
                "Error when trying to combine data across channels:\nThe "
                "lengths of the inputs do not match: 'ch_names_old' "
                f"({lengths[0]}); 'ch_names_new' ({lengths[1]}); "
                f"'ch_types_new' ({lengths[2]}); "
                f"'ch_coords_new' ({lengths[3]}); "
                f"'ch_regions_new' ({lengths[4]}); "
                f"'ch_hemispheres_new' ({lengths[5]});"
            )

    def _sort_combination_inputs_strings(
        self,
        ch_names_old: list[list[str]],
        inputs: Union[list[Union[str, None]], None],
        input_type: str,
    ) -> list[str]:
        """Sorts the inputs for combining channels that consist of a list of
        strings.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        inputs : list[str | None] | None
        -   Features of the new, combined channels.
        -   If not None and no entries are None, no changes are made.
        -   If some entries are None, the inputs are determined based on the
            channels being combined. For this, the features of the channels
            being combined should be identical.
        -   If None, all entries are automatically determined.

        input_type : str
        -   The type of input being sorted.
        -   Supported values are: 'ch_types'; 'ch_hemispheres'; and
            'ch_regions'.

        RETURNS
        -------
        inputs : list[str]
        -   The sorted features of the new, combined channels.
        """

        supported_input_types = ["ch_types", "ch_hemispheres", "ch_regions"]
        if input_type not in supported_input_types:
            raise UnavailableProcessingError(
                "Error when trying to combine data over channels:\n"
                f"The 'input_type' '{input_type}' is not recognised. "
                f"Supported values are {supported_input_types}."
            )

        if inputs is None:
            inputs = [None] * len(ch_names_old)

        for i, value in enumerate(inputs):
            if value is None:
                if input_type == "ch_types":
                    existing_values = np.unique(
                        self.data[0].get_channel_types(picks=ch_names_old[i])
                    )
                elif input_type == "ch_regions":
                    existing_values = np.unique(
                        [
                            self.extra_info["ch_regions"][channel]
                            for channel in ch_names_old[i]
                        ]
                    )
                elif input_type == "ch_hemispheres":
                    existing_values = np.unique(
                        [
                            self.extra_info["ch_hemispheres"][channel]
                            for channel in ch_names_old[i]
                        ]
                    )
                if len(existing_values) > 1:
                    raise ChannelAttributeError(
                        "Error when trying to combine data over channels:\n"
                        f"The '{input_type}' for the combination of channels "
                        f"{ch_names_old[i]} is not specified, but cannot be "
                        "automatically generated as the data is being combined "
                        f"over channels with different '{input_type}' features "
                        f"({existing_values})."
                    )
                else:
                    inputs[i] = existing_values[0]

        return inputs

    def _sort_combination_inputs_numbers(
        self,
        ch_names_old: list[list[str]],
        inputs: Union[list[Union[list[Union[int, float]], None]], None],
        input_type: str,
    ) -> list[str]:
        """Sorts the inputs for combining channels that consist of a list of
        lists of numbers.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        inputs : list[list[int | float] | None] | None
        -   Features of the new, combined channels.
        -   If not None and no entries are None, no changes are made.
        -   If some entries are None, the inputs are determined based on the
            channels being combined. For this, the features of the channels
            being combined should be identical.
        -   If None, all entries are automatically determined.

        input_type : str
        -   The type of input being sorted.
        -   Supported values are: 'ch_coords'.

        RETURNS
        -------
        ch_types_new : list[str]
        -   The sorted features of the new, combined channels.
        """

        supported_input_types = ["ch_coords"]
        if input_type not in supported_input_types:
            raise UnavailableProcessingError(
                "Error when trying to combine data over channels:\n"
                f"The 'input_type' '{input_type}' is not recognised. "
                f"Supported values are {supported_input_types}."
            )

        if inputs is None:
            inputs = [None] * len(ch_names_old)

        for i, value in enumerate(inputs):
            if value is None:
                if input_type == "ch_coords":
                    new_value = np.mean(
                        [
                            self.get_coordinates(channel)[0]
                            for channel in ch_names_old[i]
                        ],
                        axis=0,
                    ).tolist()
                inputs[i] = new_value

        return inputs

    def _sort_combination_inputs(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> tuple[list[str], list[Union[int, float]], list[str]]:
        """Sorts the inputs for combining data over channels.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        ch_names_new : list[str]
        -   The names of the new, combined channels, corresponding to the
            channel names in 'ch_names_old'.

        ch_types_new : list[str | None] | None
        -   The types of the new, combined channels.
        -   If an entry is None, the type is determined based on the types of
            the channels being combined. This only works if all channels being
            combined are of the same type.
        -   If None, all types are determined automatically.

        ch_coords_new : list[list[int | float] | None] | None
        -   The coordinates of the new, combined channels.
        -   If an entry is None, the coordinates are determined by averaging
            across the coordinates of the channels being combined.
        -   If None, the coordinates are automatically determined for all
            channels.

        ch_regions_new : list[str | None] | None
        -   The regions of the new, combined channels.
        -   If an entry is None, the region is determined based on the regions
            of the channels being combined. This only works if all channels
            being combined are from the same region.
        -   If None, all regions are determined automatically.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the new, combined channels.
        -   If an entry is None, the hemisphere is determined based on the
            hemispheres of the channels being combined. This only works if all
            channels being combined are from the same hemisphere.
        -   If None, all hemispheres are determined automatically.
        """

        self._check_combination_input_lengths(
            ch_names_old=ch_names_old,
            ch_names_new=ch_names_new,
            ch_types_new=ch_types_new,
            ch_coords_new=ch_coords_new,
            ch_regions_new=ch_regions_new,
            ch_hemispheres_new=ch_hemispheres_new,
        )

        ch_types_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_types_new,
            input_type="ch_types",
        )
        ch_regions_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_regions_new,
            input_type="ch_regions",
        )
        ch_hemispheres_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_hemispheres_new,
            input_type="ch_hemispheres",
        )
        ch_coords_new = self._sort_combination_inputs_numbers(
            ch_names_old=ch_names_old,
            inputs=ch_coords_new,
            input_type="ch_coords",
        )

        return ch_types_new, ch_coords_new, ch_regions_new, ch_hemispheres_new

    def _combine_channel_data(self, to_combine: list[list[str]]) -> NDArray:
        """Combines the data of channels through addition.

        PARAMETERS
        ----------
        to_combine : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        RETURNS
        -------
        combined_data : list[numpy array]
        -   The combined data of the channels in each sublist in 'to_combine'.
        """

        combined_data = []
        for data in self.data:
            for channels in to_combine:
                data_arr = np.sum(
                    deepcopy(data.get_data(picks=channels)), axis=0
                )
                combined_data.append(data_arr)

        return combined_data

    def add_channels(
        self,
        data: list[list[NDArray]],
        data_dimensions: list[str],
        ch_names: list[str],
        ch_types: list[str],
        ch_coords: list[list[Union[int, float]]],
        ch_reref_types: list[str],
        ch_regions: list[str],
        ch_hemispheres: list[str],
        ch_epoch_orders: Union[list[str], None] = None,
    ) -> None:
        """Adds channels to the Signal object.
        -   Data for the new channels should have the same sampling frequency as
            the data of channels already in the object.
        -   Each new channel must contain data for each window of the data in
            the signal object.

        PARAMETERS
        ----------
        data : list[list[numpy array]]
        -   List containing the data for the new channels, with each entry
            consisting of list corresponding to a data window, with each entry
            being a numpy array for the data of a channel.

        data_dimensions : list[str]
        -   Names of the dimensions in the data.

        ch_names : list[str]
        -   The names of the new channels.

        ch_types : list[str]
        -   The types of the new channels.

        ch_coords : list[list[int | float]]
        -   The coordinates of the new channels, with each entry in the list
            being a sublist containing the x-, y-, and z-axis coordinates of the
            channel.

        ch_reref_types : list[str]
        -   Rereferencing types of the new channels.

        ch_regions : list[str]
        -   The regions of the new channels.

        ch_hemispheres : list[str]
        -   The hemispheres of the new channels.

        ch_epoch_orders : list[str] | None; default None
        -   Epoch orders of the new channels. Entries should be "original" or
            "shuffled". If no epoching of the data has yet been performed, the
            value should be None (default).

        RAISES
        ------
        ValueError
        -   Raised if the number of windows in the original data and the data
            being added do not match.
        """

        if len(data) != len(self.data):
            raise ValueError(
                "Error when adding channels to the Signal object:\nThe number "
                f"of windows in the original data ({len(self.data)}) and in "
                f"the data being added ({len(data)}) do not match. If data is "
                "being added, data must be given for each window in the "
                "original data."
            )

        data = rearrange_axes(
            obj=data, old_order=data_dimensions, new_order=self.data_dimensions
        )

        for i, data_window in enumerate(data):
            new_channels, _ = create_mne_data_object(
                data=data_window,
                data_dimensions=self.data_dimensions[1:],
                ch_names=ch_names,
                ch_types=ch_types,
                sfreq=self.data[0].info["sfreq"],
                ch_coords=ch_coords,
                verbose=self._verbose,
            )
            self.data[i].add_channels([new_channels], force_update_info=True)

        for i, name in enumerate(ch_names):
            self.extra_info["ch_reref_types"][name] = ch_reref_types[i]
            self.extra_info["ch_regions"][name] = ch_regions[i]
            self.extra_info["ch_hemispheres"][name] = ch_hemispheres[i]
            if ch_epoch_orders is not None:
                self.extra_info["ch_epoch_orders"][name] = ch_epoch_orders[i]

    def combine_channels(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: Optional[list[Union[str, None]]] = None,
        ch_coords_new: Optional[
            list[Union[list[Union[int, float]], None]]
        ] = None,
        ch_regions_new: Optional[list[Union[str, None]]] = None,
        ch_hemispheres_new: Optional[list[Union[str, None]]] = None,
    ) -> None:
        """Combines the data of multiple channels in the mne.io.Raw object through
        addition and adds this combined data as a new channel.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        ch_names_new : list[str]
        -   The names of the new, combined channels, corresponding to the
            channel names in 'ch_names_old'.

        ch_types_new : list[str | None] | None; default None
        -   The types of the new, comined channels.
        -   If an entry is None, the type is determined based on the types of
            the channels being combined. This only works if all channels being
            combined are of the same type.
        -   If None, all types are determined automatically.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the new, combined channels.
        -   If an entry is None, the coordinates are determined by averaging
            across the coordinates of the channels being combined.
        -   If None, the coordinates are automatically determined for all
            channels.

        ch_regions_new : list[str | None] | None; default None
        -   The regions of the new, combined channels.
        -   If an entry is None, the region is determined based on the regions
            of the channels being combined. This only works if all channels
            being combined are from the same region.
        -   If None, all regions are determined automatically.

        ch_hemispheres_new : list[str | None] | None; default None
        -   The hemispheres of the new, combined channels.
        -   If an entry is None, the hemisphere is determined based on the
            hemispheres of the channels being combined. This only works if all
            channels being combined are from the same hemisphere.
        -   If None, all hemispheres are determined automatically.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data has been windowed or epoched.
        """

        if self._windowed:
            raise ProcessingOrderError(
                "Error when attempting to combine data across channels:\nThis "
                "is only supported for non-windowed data, however the data has "
                "been windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when attempting to combine data across channels:\nThis "
                "is only supported for non-epoched data, however the data has "
                "been epoched."
            )

        (
            ch_types_new,
            ch_coords_new,
            ch_regions_new,
            ch_hemispheres_new,
        ) = self._sort_combination_inputs(
            ch_names_old=ch_names_old,
            ch_names_new=ch_names_new,
            ch_types_new=ch_types_new,
            ch_coords_new=ch_coords_new,
            ch_regions_new=ch_regions_new,
            ch_hemispheres_new=ch_hemispheres_new,
        )

        combined_data = self._combine_channel_data(
            to_combine=ch_names_old,
        )

        self.add_channels(
            data=combined_data,
            data_dimensions=["channels", "timepoints"],
            ch_names=ch_names_new,
            ch_types=ch_types_new,
            ch_coords=ch_coords_new,
            ch_reref_types=[None] * len(ch_names_new),
            ch_regions=ch_regions_new,
            ch_hemispheres=ch_hemispheres_new,
            ch_epoch_orders=None,
        )

        if self._verbose:
            print(
                "Creating new channels of data by combining the data of "
                "pre-existing channels:"
            )
            [
                print(f"{ch_names_old[i]} -> {ch_names_new[i]}")
                for i in range(len(ch_names_old))
            ]
            print("\n")

    def drop_unrereferenced_channels(self) -> None:
        """Drops channels that have not been rereferenced from the mne.io.Raw or
        mne.Epochs object, also discarding entries for these channels from
        'extra_info'.
        """

        self._drop_channels(
            [
                ch_name
                for ch_name in self.extra_info["ch_reref_types"].keys()
                if self.extra_info["ch_reref_types"][ch_name] == "none"
            ]
        )

    def _apply_rereference(
        self,
        reref_method: Reref,
        ch_names_old: list[Union[str, list[str]]],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> tuple[mne.io.Raw, list[str], dict[str], dict[str]]:
        """Applies a rereferencing method to the mne.io.Raw object.

        PARAMETERS
        ----------
        reref_method : Reref
        -   The rereferencing method to apply.

        ch_names_old : list[str | list[str]]
        -   The names of the channels in the mne.io.Raw object to rereference.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

        ch_names_new : list[str | None] | None
        -   The names of the newly rereferenced channels, corresponding to the
            channels used for rerefrencing in ch_names_old.
        -   If some or all entries are None, names of the new channels are
            determined based on those they are referenced from.

        ch_types_new : list[str | None] | None
        -   The types of the newly rereferenced channels as recognised by MNE,
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, types of the new channels are
            determined based on those they are referenced from.

        reref_types : list[str | None] | None
        -   The rereferencing type applied to the channels, corresponding to the
            channels in 'ch_names_new'.
        -   If some or all entries are None, types of the new channels are
            determined based on those they are referenced from.

        ch_coords_new : list[list[int | float] | None] | None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is None, the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some entries are None, those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels channels, corresponding to
            the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        RETURNS
        -------
        mne.io.Raw
        -   The rereferenced data in an mne.io.Raw object.

        list[str]
        -   Names of the channels that were produced by the rereferencing.

        dict[str]
        -   Dictionary showing the rereferencing types applied to the channels,
            in which the key:value pairs are channel name : rereference type.

        dict[str]
        -   Dictionary showing the regions of the rereferenced channels, in
            which the key:value pairs are channel name : region.
        """

        return reref_method(
            deepcopy(self.data[0]),
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_hemispheres_new,
        ).rereference()

    def _remove_conflicting_channels(self, ch_names: list[str]) -> None:
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
            "Removing the channels from the raw data.\n"
        )

    def _append_rereferenced_raw(self, rerefed_raw: mne.io.Raw) -> None:
        """Appends a rereferenced mne.io.Raw object to the self mne.io.Raw
        object, first discarding channels in the self mne.io.Raw object which
        have the same names as those in the mne.io.Raw object to append.

        PARAMETERS
        ----------
        rerefed_raw : mne.io.Raw
        -   An mne.io.Raw object that has been rereferenced which will be
            appended.
        """

        _, repeated_chs = check_repeated_vals(
            to_check=[
                *self.data[0].info["ch_names"],
                *rerefed_raw.info["ch_names"],
            ]
        )
        if repeated_chs is not None:
            self._remove_conflicting_channels(repeated_chs)

        self.data[0].add_channels([rerefed_raw])

    def _add_rereferencing_info(self, info_to_add: dict) -> None:
        """Adds channel rereferencing information to 'extra_info'.

        PARAETERS
        ---------
        info_to_add : dict
        -   A dictionary used for updating 'extra_info'.
        -   The dictionary's keys are the names of the entries that will be
            updated in 'extra_info' with the corresponding values.
        """

        [
            self.extra_info[key].update(info_to_add[key])
            for key in info_to_add.keys()
        ]

    def _get_channel_rereferencing_pairs(
        self,
        ch_names_old: list[Union[str, list[str]]],
        ch_names_new: list[Union[str, list[str]]],
    ) -> list[str]:
        """Collects the names of the channels that were referenced and the newly
        generated channels together.

        PARAMETERS
        ----------
        ch_names_old : list[str | list[str]]
        -   Names of the channels that were rereferenced.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

        ch_names_new : list[str]
        -   Names of the channels that were produced by the rereferencing.

        RETURNS
        -------
        list[str | list[str]]
        -   List of sublists, in which each sublist contains the name(s) of the
            channel(s) that was(were) rereferenced, and the name of the channel
            that was produced.
        """

        return [
            [ch_names_old[i], ch_names_new[i]] for i in range(len(ch_names_old))
        ]

    def _rereference(
        self,
        reref_method: Reref,
        ch_names_old: list[Union[str, list[str]]],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> list[str]:
        """Parent method for calling on other methods to rereference the data,
        add it to the self mne.io.Raw object, and add the rereferecing
        information to 'extra_info'.

        PARAMETERS
        ----------
        RerefMethod : Reref
        -   The rereferencing method to apply.

        ch_names_old : list[str]
        -   The names of the channels in the mne.io.Raw object to rereference.

        ch_names_new : list[str | None] | None; default None
        -   The names of the newly rereferenced channels, corresponding to the
            channels used for rerefrencing in ch_names_old.
        -   Missing values (None) will be set based on 'ch_names_old'.

        ch_types_new : list[str | None] | None; default None
        -   The types of the newly rereferenced channels as recognised by MNE,
            corresponding to the channels in 'ch_names_new'.
        -   Missing values (None) will be set based on the types of channels in
            'ch_names_old'.

        ch_reref_types : list[str | None] | None; default None
        -   The rereferencing type applied to the channels, corresponding to the
            channels in 'ch_names_new'.
        -   Missing values (None) will be set as 'common_average'.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels channels, corresponding to
            the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        RETURNS
        -------
        ch_names_new : list[str]
        -   List containing the names of the new, rereferenced channels.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to rereference the data after it has
            already been windowed or epoched.
        """

        if self._windowed:
            raise ProcessingOrderError(
                "Error when rereferencing the data:\nThe data to rereference "
                "has been windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when rereferencing the data:\nThe data to rereference "
                "has been epoched."
            )

        (
            rerefed_raw,
            ch_names_new,
            ch_reref_types_dict,
            ch_regions_dict,
            ch_hemispheres_dict,
        ) = self._apply_rereference(
            reref_method,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_hemispheres_new,
        )
        self._append_rereferenced_raw(rerefed_raw)
        self._add_rereferencing_info(
            info_to_add={
                "ch_reref_types": ch_reref_types_dict,
                "ch_regions": ch_regions_dict,
                "ch_hemispheres": ch_hemispheres_dict,
            }
        )

        self._rereferenced = True

        return ch_names_new

    def rereference_bipolar(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> None:
        """Bipolar rereferences channels in the mne.io.Raw object.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   The names of the channels in the mne.io.Raw object to rereference.

        ch_names_new : list[str | None] | None; default None
        -   The names of the newly rereferenced channels, corresponding to the
            channels used for rerefrencing in ch_names_old.
        -   Missing values (None) will be set based on 'ch_names_old'.

        ch_types_new : list[str | None] | None; default None
        -   The types of the newly rereferenced channels as recognised by MNE,
            corresponding to the channels in 'ch_names_new'.
        -   Missing values (None) will be set based on the types of channels in
            'ch_names_old'.

        ch_reref_types : list[str | None] | None; default None
        -   The rereferencing type applied to the channels, corresponding to the
            channels in 'ch_names_new'.
        -   Missing values (None) will be set as 'common_average'.

        ch_coords_new : list[list[int | float] | None] or None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels channels, corresponding to
            the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.
        """

        ch_names_new = self._rereference(
            RerefBipolar,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_hemispheres_new,
        )

        self._rereferenced_bipolar = True
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps("rereferencing_bipolar", ch_reref_pairs)
        if self._verbose:
            print("The following channels have been bipolar rereferenced:")
            [
                print(f"{old[0]} - {old[1]} -> {new}")
                for [old, new] in ch_reref_pairs
            ]
            print("\n")

    def rereference_common_average(
        self,
        ch_names_old: list[str],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> None:
        """Common-average rereferences channels in the mne.io.Raw object.

        PARAMETERS
        ----------
        ch_names_old : list[str]
        -   The names of the channels in the mne.io.Raw object to rereference.

        ch_names_new : list[str | None] | None; default None
        -   The names of the newly rereferenced channels, corresponding to the
            channels used for rerefrencing in ch_names_old.
        -   Missing values (None) will be set based on 'ch_names_old'.

        ch_types_new : list[str | None] | None; default None
        -   The types of the newly rereferenced channels as recognised by MNE,
            corresponding to the channels in 'ch_names_new'.
        -   Missing values (None) will be set based on the types of channels in
            'ch_names_old'.

        ch_reref_types : list[str | None] | None; default None
        -   The rereferencing type applied to the channels, corresponding to the
            channels in 'ch_names_new'.
        -   Missing values (None) will be set as 'common_average'.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels channels, corresponding to
            the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.
        """

        ch_names_new = self._rereference(
            RerefCommonAverage,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_hemispheres_new,
        )

        self._rereferenced_common_average = True
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps(
            "rereferencing_common_average", ch_reref_pairs
        )
        if self._verbose:
            print(
                "The following channels have been common-average rereferenced:"
            )
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]
            print("\n")

    def rereference_pseudo(
        self,
        ch_names_old: list[str],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: list[str],
        ch_coords_new: Optional[list[Optional[list[Union[int, float]]]]],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
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

        ch_names_new : list[str | None] | None; default None
        -   The names of the newly rereferenced channels, corresponding to the
            channels used for rerefrencing in ch_names_old.
        -   Missing values (None) will be set based on 'ch_names_old'.

        ch_types_new : list[str | None] | None; default None
        -   The types of the newly rereferenced channels as recognised by MNE,
            corresponding to the channels in 'ch_names_new'.
        -   Missing values (None) will be set based on the types of channels in
            'ch_names_old'.

        ch_reref_types : list[str]
        -   The rereferencing type applied to the channels, corresponding to the
            channels in 'ch_names_new'.
        -   No missing values (None) can be given, as the rereferencing type
            cannot be determined dynamically from this arbitrary rereferencing
            method.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels channels, corresponding to
            the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.
        """

        ch_names_new = self._rereference(
            RerefPseudo,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_hemispheres_new,
        )

        self._rereferenced_pseudo = True
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            ch_names_old, ch_names_new
        )
        self._update_processing_steps("rereferencing_pseudo", ch_reref_pairs)
        if self._verbose:
            print("The following channels have been pseudo rereferenced:")
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]

    def window(self, window_length: Union[int, float]) -> None:
        """Converts an MNE Raw object into multiple MNE Raw objects, each with a
        specified duration.

        PARAMETERS
        ----------
        window_length : int | float
        -   The duration of the windows (seconds) to divide the data into.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to divide the data into windows once it
            has already been windowed or epoched.
        """

        if self._windowed:
            raise ProcessingOrderError(
                "Error when windowing the data:\nThe data has already been "
                "windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when windowing the data:\nThe data has been epoched. "
                "Windowing can only be performed on non-epoched data."
            )

        windows = mne.make_fixed_length_epochs(self.data[0], window_length)
        windowed_data = []
        for window in windows:
            data, _ = create_mne_data_object(
                data=window,
                data_dimensions=["channels", "timepoints"],
                ch_names=windows.ch_names,
                ch_types=windows.get_channel_types(),
                sfreq=windows.info["sfreq"],
                ch_coords=self.get_coordinates(),
                subject_info=windows.info["subject_info"],
            )
            windowed_data.append(data)
        self.data = windowed_data

        self._windowed = True
        self._update_processing_steps("window_data", window_length)
        if self._verbose:
            print(
                f"Windowing the data with window lengths of {window_length} "
                "seconds.\n"
            )

    def epoch(self, epoch_length: int) -> None:
        """Converts the data in one or many MNE Raw object(s) into one or many
        MNE Epochs object(s) containing epochs of a specified duration.

        PARAMETERS
        ----------
        epoch_length : int | float
        -   The duration of the epochs (seconds) to divide the data into.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to epoch the data once it has already
            been epoched.
        -   This method can only be called if the data is stored as an MNE Raw
            object, not as an MNE Epochs object.
        """

        if self._epoched:
            raise ProcessingOrderError(
                "Error when epoching the data:\nThe data has already been "
                "epoched."
            )

        for i, data in enumerate(self.data):
            self.data[i] = mne.make_fixed_length_epochs(data, epoch_length)
            self.data[i].load_data()

        self._epoched = True
        self._update_processing_steps("epoch_data", epoch_length)
        self.data_dimensions = ["windows", "epochs", "channels", "timepoints"]
        self.extra_info["ch_epoch_orders"] = {
            name: "original" for name in self.data[0].ch_names
        }
        if self._verbose:
            print(
                f"Epoching the data with epoch lengths of {epoch_length} "
                "seconds.\n"
            )

    def shuffle(
        self,
        channels: list[str],
        n_shuffles: int,
        rng_seed: Union[int, float, None] = None,
    ) -> None:
        """Creates new channels by randomly reordering the epochs of channels,
        creating time-series data with a disrupted temporal order.

        PARAMETERS
        ----------
        channels : list[str]
        -   Names of the channels to create shuffled copies of.

        n_shuffles : int
        -   The number of shuffled copies to create for each channel.

        rng_seed : int | float
        -   The seed to use for the random number generator. The seed is set
            once before any shuffling takes place, and is not set again once the
            shuffling has begun.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data has not been epoched.
        """

        if not self._epoched:
            raise ProcessingOrderError(
                "Error when shuffling the data:\nThe data has not been "
                "epoched, but shuffling requires the data be epoched."
            )

        if rng_seed is not None:
            np.random.seed(rng_seed)

        shuffled_data = []
        for window_i, data in enumerate(self.data):
            to_shuffle = deepcopy(data)
            to_shuffle.pick_channels(ch_names=channels, ordered=True)
            shuffled_data.append([])
            for shuffle_i in range(n_shuffles):
                epoch_order = np.arange(len(to_shuffle.events))
                np.random.shuffle(epoch_order)
                shuffled_data[window_i].append(
                    concatenate_epochs(
                        [to_shuffle[epoch_order]], verbose="ERROR"
                    )
                )
                shuffled_data[window_i][shuffle_i].rename_channels(
                    {name: f"SHUFFLED[{shuffle_i}]_{name}" for name in channels}
                )
            if n_shuffles > 1:
                shuffled_data[window_i] = shuffled_data[window_i][
                    0
                ].add_channels(shuffled_data[window_i][1:])
            else:
                shuffled_data[window_i] = shuffled_data[window_i][0]

        ch_reref_types = (
            ordered_list_from_dict(channels, self.extra_info["ch_reref_types"])
            * n_shuffles
        )
        ch_regions = (
            ordered_list_from_dict(channels, self.extra_info["ch_regions"])
            * n_shuffles
        )
        ch_hemispheres = (
            ordered_list_from_dict(channels, self.extra_info["ch_hemispheres"])
            * n_shuffles
        )
        self.add_channels(
            data=shuffled_data,
            data_dimensions=self.data_dimensions,
            ch_names=shuffled_data[0].ch_names,
            ch_types=shuffled_data[0].get_channel_types(),
            ch_coords=shuffled_data[0]._get_channel_positions(),
            ch_reref_types=ch_reref_types,
            ch_regions=ch_regions,
            ch_hemispheres=ch_hemispheres,
            ch_epoch_orders=["shuffled"] * len(channels) * n_shuffles,
        )

        self._shuffled = True
        self._update_processing_steps(
            "shuffle_data",
            {
                "channels": channels,
                "n_shuffles": n_shuffles,
                "rng_seed": rng_seed,
            },
        )
        for n_shuffle in range(n_shuffles):
            for name in shuffled_data[n_shuffle].ch_names:
                self.extra_info["ch_epoch_orders"][name] = "shuffled"
        if self._verbose:
            print(
                "Creating epoch-shuffled data for the following channels over "
                f"{n_shuffles} iterations:\n{channels}\n"
            )

    def _extract_data(self, rearrange: Union[list[str], None]) -> NDArray:
        """Extracts the signals from the mne.io.Raw object.

        PARAMETERS
        ----------
        rearrange : list[str] | None; default None
        -   How to rearrange the axes of the data once extracted.
        -   E.g. ["channels", "epochs", "timepoints"] would give data in the
            format channels x epochs x timepoints
        -   If None, the data is taken as is.

        RETURNS
        -------
        extracted_data : numpy array
        -   The time-series signals extracted from the mne.io.Raw oject.
        """

        extracted_data = []
        for data in self.data:
            extracted_data.append(deepcopy(data.get_data()))
        extracted_data = np.asarray(extracted_data)

        if rearrange is not None:
            extracted_data = rearrange_axes(
                obj=extracted_data,
                old_order=self.data_dimensions,
                new_order=rearrange,
            )
            self.data_dimensions = rearrange

        return extracted_data

    def save_object(
        self, fpath: str, ask_before_overwrite: Optional[bool] = None
    ) -> None:
        """Saves the Signal object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """

        if not check_ftype_present(fpath):
            fpath += ".pkl"

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose
        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            save_as_pkl(to_save=self, fpath=fpath)

    def data_as_dict(self) -> dict:
        """Returns the data as a dictionary.

        RETURNS
        -------
        data_dict : dict
        -   The data as a dictionary.
        """

        extracted_data = self._extract_data(
            rearrange=["channels", "windows", "epochs", "timepoints"]
        ).tolist()

        data_dict = {
            "data": extracted_data,
            "data_dimensions": self.data_dimensions,
            "ch_names": self.data[0].ch_names,
            "ch_types": self.data[0].get_channel_types(),
            "ch_coords": self.get_coordinates(),
            "ch_regions": ordered_list_from_dict(
                self.data[0].ch_names, self.extra_info["ch_regions"]
            ),
            "ch_hemispheres": ordered_list_from_dict(
                self.data[0].ch_names, self.extra_info["ch_hemispheres"]
            ),
            "ch_reref_types": ordered_list_from_dict(
                self.data[0].ch_names, self.extra_info["ch_reref_types"]
            ),
            "samp_freq": self.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.data[0].info["subject_info"],
        }
        if self._epoched:
            data_dict["ch_epoch_orders"] = ordered_list_from_dict(
                self.data[0].ch_names, self.extra_info["ch_epoch_orders"]
            )

        return data_dict

    def save_as_dict(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the time-series data and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved.

        ftype : str | None; default None
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if the given format for saving the file is in an unsupported
            format.
        """

        to_save = self.data_as_dict()

        if ftype is None:
            ftype = identify_ftype(fpath)
        if not check_ftype_present(fpath):
            fpath += ftype

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose
        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            if ftype == "json":
                save_as_json(to_save=to_save, fpath=fpath)
            elif ftype == "pkl":
                save_as_pkl(to_save=to_save, fpath=fpath)
            else:
                raise UnavailableProcessingError(
                    f"Error when trying to save the raw signals:\nThe {ftype} "
                    "format for saving is not supported."
                )
            if self._verbose:
                print(f"Saving the raw signals to:\n'{fpath}'.\n")


def data_dict_to_signal(data: dict) -> Signal:
    """Converts a data dictionary into a Signal object.
    -   The signals themselves will be converted to either an MNE Raw object or
        MNE Epochs object, depending on whether the signals have been epoched.

    PARAMETERS
    ----------
    data : dict
    -   Data dictionary, following the same structure as a data dictionary
        derived from a Signal object.

    RETURNS
    -------
    signal : Signal
    -   The data dictionary as a Signal object.
    """

    signal = Signal()

    if "epochs" in data["data_dimensions"]:
        new_dims = ["windows", "epochs", "channels", "timepoints"]
    else:
        new_dims = ["windows", "channels", "timepoints"]
    data_windows = rearrange_axes(
        obj=data["data"],
        old_order=data["data_dimensions"],
        new_order=new_dims,
    )

    mne_objects = []
    for data_window in data_windows:
        mne_object, _ = create_mne_data_object(
            data=data_window,
            data_dimensions=new_dims[1:],
            ch_names=data["ch_names"],
            ch_types=data["ch_types"],
            ch_coords=data["ch_coords"],
            sfreq=data["samp_freq"],
            subject_info=data["subject_info"],
        )
        mne_objects.append(mne_object)

    extra_info = create_extra_info(data=data)

    signal.data_from_objects(
        data=mne_objects,
        data_dimensions=new_dims,
        processing_steps=data["processing_steps"],
        extra_info=extra_info,
    )

    return signal
