"""Classes for calculating connectivity between signals.

CLASSES
-------
ConnectivityCoherence : subclass of the abstract base class 'ProcMethod'
-   Calculates the coherence (standard or imaginary) between signals.
"""

from copy import deepcopy
from typing import Optional, Union
from mne import concatenate_epochs
from mne_connectivity import (
    seed_target_indices,
    spectral_connectivity_epochs,
    SpectroTemporalConnectivity,
)
from numpy.typing import NDArray
import numpy as np
from coh_exceptions import (
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_handle_entries import (
    FillerObject,
    ordered_list_from_dict,
    rearrange_axes,
)
from coh_processing_methods import ProcMethod
import coh_signal
from coh_saving import save_object, save_dict


class ConnectivityCoherence(ProcMethod):
    """Calculates the coherence (standard or imaginary) between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs coherence analysis.

    save_object
    -   Saves the ConnectivityCoherence object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)

        # Initialises inputs of the object.
        self._sort_inputs()

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self.coherence = None
        self.coherence_dims = None
        self._method = None
        self._seeds = None
        self._targets = None
        self._indices = None
        self._mode = None
        self._fmin = None
        self._fmax = None
        self._fskip = None
        self._faverage = None
        self._tmin = None
        self._tmax = None
        self._mt_bandwidth = None
        self._mt_adaptive = None
        self._mt_low_bias = None
        self._cwt_freqs = None
        self._cwt_n_cycles = None
        self._shuffle_group = None
        self._n_shuffles = None
        self._shuffle_rng_seed = None
        self._average_timepoints = None
        self._block_size = None
        self._n_jobs = None
        self._shuffled_channels = None

        # Initialises aspects of the object that indicate which methods have
        # been called (starting as 'False'), which can later be updated.
        self._shuffled_present = False
        self._shuffled_sorted = False
        self._timepoints_averaged = False

    def _sort_inputs(self) -> None:
        """Checks the inputs to the object to ensure that they match the
        requirements for processing and assigns inputs.

        RAISES
        ------
        ValueError
        -   Raised if the dimensions of the data in the Signal object is not
            supported.
        """

        supported_data_dims = ["epochs", "channels", "timepoints"]
        if self.signal.data_dimensions != supported_data_dims:
            raise ValueError(
                "Error when trying to perform coherence analysis on the "
                "data:\nData in the Signal object has the dimensions "
                f"{self.signal.data_dimensions}, but only data with dimensions "
                f"{supported_data_dims} is supported."
            )

        super()._sort_inputs()

        self.extra_info["epoch_orders"] = {
            ch_name: "original" for ch_name in self.signal.data.ch_names
        }

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""

        self._indices = seed_target_indices(
            seeds=[
                i
                for i, name in enumerate(self.signal.data.ch_names)
                if name in self._seeds
            ],
            targets=[
                i
                for i, name in enumerate(self.signal.data.ch_names)
                if name in self._targets
            ],
        )

    def _sort_indices(self) -> None:
        """Sorts the inputs for generating MNE-readable indices for calculating
        connectivity between signals."""

        groups = ["_seeds", "_targets"]
        channel_types = self.signal.data.get_channel_types()
        for group in groups:
            channels = getattr(self, group)
            if channels is None:
                setattr(self, group, self.signal.data.ch_names)
            elif isinstance(channels, str):
                if channels[:5] == "type_":
                    setattr(
                        self,
                        group,
                        [
                            name
                            for i, name in enumerate(self.signal.data.ch_names)
                            if channel_types[i] == channels[5:]
                        ],
                    )
                else:
                    setattr(self, group, [channels])

        self._generate_indices()

    def _generate_shuffled_data(self) -> None:
        """Generates time-series data for a set of channels in which the order
        of epochs is randomly organised (i.e. 'shuffled') and adds it to the
        data."""

        shuffle_channels = getattr(self, f"_{self._shuffle_group}")
        data_to_shuffle = self.signal.data.copy()
        data_to_shuffle = data_to_shuffle.pick_channels(
            ch_names=shuffle_channels, ordered=True
        )
        if self._shuffle_rng_seed is not None:
            np.random.seed(self._shuffle_rng_seed)
        shuffled_data = []
        shuffled_channels = {name: [] for name in shuffle_channels}
        for shuffle_n in range(self._n_shuffles):
            epoch_order = np.arange(len(data_to_shuffle.events))
            np.random.shuffle(epoch_order)
            shuffled_data.append(
                concatenate_epochs(
                    [data_to_shuffle[epoch_order]], verbose=False
                )
            )
            shuffled_data[shuffle_n].rename_channels(
                {
                    name: f"SHUFFLED[{shuffle_n}]_{name}"
                    for name in shuffle_channels
                }
            )
            [
                shuffled_channels[name].append(f"SHUFFLED[{shuffle_n}]_{name}")
                for name in shuffle_channels
            ]

        self.signal.data.add_channels(shuffled_data)
        self._shuffled_channels = shuffled_channels

        if self._verbose:
            print(
                "Creating epoch-shuffled data for the following channels:\n"
                f"{shuffle_channels}"
            )

    def _add_shuffled_to_indices(self) -> None:
        """Adds newly-created epoch-shuffled channels of data to the
        connectivity indices."""

        shuffled_channel_names = []
        [
            shuffled_channel_names.extend(value)
            for value in self._shuffled_channels.values()
        ]
        if self._shuffle_group == "seeds":
            self._seeds.extend(shuffled_channel_names)
        else:
            self._targets.extend(shuffled_channel_names)

        self._generate_indices()

    def _sort_shuffles(self) -> None:
        """Organises the shuffling of time-series data for a set of channels in
        which the order of epochs is randomly organised (i.e. 'shuffled'),
        adding these new channels to the data and updating the connectivity
        indices.
        """

        self._generate_shuffled_data()
        self._add_shuffled_to_indices()

        self._shuffled_present = True

    def _sort_processing_inputs(self) -> None:
        """Converts the connectivity seeds and targets into channel indices for
        the connectivity analysis, and generates epoch-shuffled data, if
        requested."""

        self._sort_indices()

        if self._cwt_freqs is not None:
            self._cwt_freqs = np.arange(
                start=self._cwt_freqs[0], stop=self._cwt_freqs[1] + 1
            )

        if self._shuffle_group is not None:
            self._sort_shuffles()

    def _find_channel_indices_in_names(
        self, channel_names: list[str]
    ) -> tuple[dict[int], dict[list[int]]]:
        """Finds the indices of channels in the connectivity results, and groups
        the indices of channels containing shuffled data derived from the same
        channel.

        PARAMETERS
        ----------
        channel_names : list[str]
        -   The names of channels in the order they appear in the connectivity
            results whose indices should be found.

        RETURNS
        -------
        all_indices : dict[int]
        -   Dictionary containing the indices of all channels in the
            connectivity analysis, where the keys are the channel names and the
            values are the indices of these channels in the connectivty results.

        shuffled_indices : dict[list[int]]
        -   Dictionary containing the indices of the shuffled channels derived
            from the same channels, where the keys are the original channel
            names and the values are the indices of shuffled channels derived
            from these original channels.
        """

        all_indices = {name: i for i, name in enumerate(channel_names)}
        shuffled_indices = {}
        for real_channel, shuffled_channels in self._shuffled_channels.items():
            shuffled_indices[real_channel] = []
            for shuffled_channel in shuffled_channels:
                shuffled_indices[real_channel].append(
                    all_indices[shuffled_channel]
                )

        return all_indices, shuffled_indices

    def _find_shuffled_indices_in_nodes(
        self,
        results: SpectroTemporalConnectivity,
        shuffled_indices: dict[list[int]],
    ) -> dict[dict[list[int]]]:
        """Finds the indices in the connectivity results for each non-shuffled
        group channel to shuffled channel derived from the same original channel
        combination.

        PARAMETERS
        ----------
        results : MNE-Connectivity SpectroTemporalConnectivity
        -   The connectivity results containing data from epoch-shuffled
            channels.

        shuffled_indices : dict[list[int]]
        -   Dictionary containing the indices of the shuffled channels derived
            from the same channels, where the keys are the original channel
            names and the values are the indices of shuffled channels derived
            from these original channels.

        RETURNS
        -------
        connectivity_indices : dict[dict[list[int]]]
        -   Dictionary containing the indices of the results for each
            non-shuffled group channel to shuffled channel derived from the same
            original channel combination.
        -   The keys are the names of the original target channels in the data
            from which the shuffled channels are derived, with the values being
            dictionaries whose keys are the indices of the non-shuffled group
            channels connectivity is being computed for, where the values are
            lists of indices of the connectivity nodes where connectivity is
            being calculated between these non-shuffled group channels and the
            shuffled channels derived from the same original channel.
        """

        if self._shuffle_group == "seeds":
            shuffled_group_i = 0
            nonshuffled_group_i = 1
        else:
            shuffled_group_i = 1
            nonshuffled_group_i = 0

        connectivity_indices = {}
        for shuffled_name, shuffled_is in shuffled_indices.items():
            connectivity_indices[shuffled_name] = {
                i: [] for i in np.unique(results.indices[nonshuffled_group_i])
            }
            for node_i, nonshuffled_i in enumerate(
                results.indices[nonshuffled_group_i]
            ):
                if results.indices[shuffled_group_i][node_i] in shuffled_is:
                    connectivity_indices[shuffled_name][nonshuffled_i].append(
                        node_i
                    )

        return connectivity_indices

    def _sort_shuffled_nodes_names(self, old_names: list[str]) -> list[str]:
        """Replaces the names of epoch-shuffled channels being averaged over
        with a single, generic epoch-shuffled name.
        -   E.g. if the epoch-shuffled channel names were 'SHUFFLED[0]_LFP_1'
            and 'SHUFFLED[1]_LFP_1', this would be replaced with
            'SHUFFLED_LFP_1'.

        PARAMETERS
        ----------
        old_names : list[str]
        -   The names of channels in the data that will be sorted.

        RETURNS
        -------
        new_names : list[str]
        -   The sorted channel names.
        """

        new_names = []
        drop_names = []
        replace_names = {}

        for (
            original_channel,
            shuffled_channels,
        ) in self._shuffled_channels.items():
            drop_names.extend(shuffled_channels[1:])
            replace_names[shuffled_channels[0]] = original_channel

        for name in old_names:
            if name not in drop_names:
                if name in replace_names.keys():
                    new_names.append(f"SHUFFLED_{replace_names[name]}")
                else:
                    new_names.append(name)

        return new_names

    def _generate_shuffled_node_sort_indices(
        self, indices: dict[dict[list[int]]]
    ) -> tuple[list[int], list[list[int]]]:
        """Generates indices for sorting the results containing epoch-shuffled
        data.

        PARAMETERS
        ----------
        indices : dict[dict[list[int]]]
        -   Dictionary containing the indices of the results for each
            non-shuffled group channel to shuffled channel derived from the same
            original channel combination.
        -   The keys are the names of the original target channels in the data
            from which the shuffled channels are derived, with the values being
            dictionaries whose keys are the indices of the non-shuffled group
            channels connectivity is being computed for, where the values are
            lists of indices of the connectivity nodes where connectivity is
            being calculated between these non-shuffled group channels and the
            shuffled channels derived from the same original channel.

        RETURNS
        -------
        drop_node_indices : list[int]
        -   Entries to drop from the results indices and data corresponding to
            epoch-shuffled channels derived from the same original channels
            whose data is being averaged over.

        average_node_indices : list[list[int]]
        -   List of sublists containing indices of the entries to average the
            data of the epoch-shuffled channels derived from the same original
            channels.
        """

        drop_node_indices = []
        average_node_indices = []
        for nonshuffled_channels in indices.values():
            for shuffled_node_is in nonshuffled_channels.values():
                drop_node_indices.extend(shuffled_node_is[1:])
                average_node_indices.append(shuffled_node_is)

        return drop_node_indices, average_node_indices

    def _sort_shuffled_nodes_indices(
        self, old_indices: list[list[int]], drop_node_indices: list[int]
    ) -> list[list[int]]:
        """Discards entries in the connectivity indices corresponding to
        epoch-shuffled channels that are being averaged over.

        PARAMETERS
        ----------
        old_indices : list[list[int]]
        -   The indices that will be sorted in a list with dimensions [2 x
            n_nodes].

        drop_node_indices : list[int]
        -   Indices of excess entries belonging to epoch-shuffled channels
            derived from the same original channels to drop from the
            connectivity indices.

        RETURNS
        -------
        new_indices : list[list[int]]
        -   The sorted indices.
        """

        new_indices = [None, None]

        for group_i in range(2):
            new_indices[group_i] = [
                channel_i
                for entry_i, channel_i in enumerate(old_indices[group_i])
                if entry_i not in drop_node_indices
            ]

        return new_indices

    def _sort_shuffled_nodes_data(
        self,
        old_data: NDArray,
        drop_node_indices: list[int],
        average_node_indices: list[list[int]],
    ) -> NDArray:
        """Averages over data from non-shuffled group channels to epoch-shuffled
        channels derived from the same original channels.

        PARAMETERS
        ----------
        old_data : numpy array
        -   The data to be sorted.

        drop_node_indices : list[int]
        -   Indices of excess entries belonging to epoch-shuffled channels
            derived from the same original channels to drop from the
            connectivity indices.

        average_node_indices : list[list[int]]
        -   List containing sublists of indices of connectivity nodes to average
            over.

        RETURNS
        -------
        new_data : numpy array
        -   The sorted data.
        """

        for average_node_is in average_node_indices:
            old_data[average_node_is[0]] = np.mean(
                old_data[average_node_is], axis=0
            )

        new_data = [
            data
            for index, data in enumerate(old_data)
            if index not in drop_node_indices
        ]

        return np.asarray(new_data)

    def _sort_shuffled_nodes(
        self,
        results: SpectroTemporalConnectivity,
        indices: dict[dict[list[int]]],
    ) -> FillerObject:
        """Sorts the connectivity results for non-shuffled group channels to
        shuffled channels derived from the same original channels, averaging the
        results, as well as updating the channel names and connectivity indices.

        PARAMETERS
        ----------
        results : MNE-Connectivity SpectroTemporalConnectivity
        -   The connectivity results containing data from epoch-shuffled
            channels.

        indices : dict[dict[list[int]]]
        -   Dictionary containing the indices of the results for each
            non-shuffled group channel to shuffled channel derived from the same
            original channel combination.
        -   The keys are the names of the original target channels in the data
            from which the shuffled channels are derived, with the values being
            dictionaries whose keys are the indices of the non-shuffled group
            channels connectivity is being computed for, where the values are
            lists of indices of the connectivity nodes where connectivity is
            being calculated between these non-shuffled group channels and the
            shuffled channels derived from the same original channel.

        RETURNS
        -------
        new_results : FillerObject
        -   The connectivity results averaged across shuffled channels derived
            from the same original channel with the corresponding connectivity
            indices.
        """

        new_results = FillerObject()

        (
            drop_node_indices,
            average_node_indices,
        ) = self._generate_shuffled_node_sort_indices(indices=indices)

        new_results.names = self._sort_shuffled_nodes_names(
            old_names=results.names
        )
        new_results.indices = self._sort_shuffled_nodes_indices(
            old_indices=results.indices, drop_node_indices=drop_node_indices
        )
        new_results._data = self._sort_shuffled_nodes_data(
            old_data=deepcopy(results.get_data()),
            drop_node_indices=drop_node_indices,
            average_node_indices=average_node_indices,
        )
        new_results.freqs = results.freqs

        if self._verbose:
            print(
                "Averaging the epoch-shuffled results across channels derived "
                "the same parent channels."
            )

        return new_results

    def _update_extra_info_with_shuffled(self) -> None:
        """Adds information about epoch-shuffled channels to 'extra_info' based
        on the original channels the shuffled channels are derived from."""

        for name in self._shuffled_channels.keys():
            shuffled_name = f"SHUFFLED_{name}"
            self.extra_info["reref_types"][shuffled_name] = self.extra_info[
                "reref_types"
            ][name]
            self.extra_info["ch_regions"][shuffled_name] = self.extra_info[
                "ch_regions"
            ][name]
            self.extra_info["ch_hemispheres"][shuffled_name] = self.extra_info[
                "ch_hemispheres"
            ][name]
            self.extra_info["epoch_orders"][shuffled_name] = "shuffled"

    def _sort_shuffled_results(
        self, results: SpectroTemporalConnectivity
    ) -> SpectroTemporalConnectivity:
        """Sorts the connectivity results derived from epoch-shuffled data such
        that results derived from the same original channel in the data are
        averaged across to give a single shuffled connectivity result per
        original channel.

        PARAMETERS
        ----------
        results : MNE-Connectivity SpectroTemporalConnectivity
        -   The connectivity results containing data from epoch-shuffled
            channels.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the epoch-shuffled data has already been sorted.
        """

        if self._shuffled_sorted:
            raise ProcessingOrderError(
                "Error when sorting the epoch-shuffled data: The data has "
                "already been sorted."
            )

        (
            _,
            shuffled_indices,
        ) = self._find_channel_indices_in_names(channel_names=results.names)

        connectivity_indices = self._find_shuffled_indices_in_nodes(
            results=results,
            shuffled_indices=shuffled_indices,
        )

        results = self._sort_shuffled_nodes(
            indices=connectivity_indices, results=results
        )

        self._update_extra_info_with_shuffled()

        self._shuffled_sorted = True

        return results

    def _average_timepoints_results(self) -> None:
        """Averages the connectivity results across timepoints.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the timepoints have already been averaged across.

        UnavailableProcessingError
        -   Raised if timepoints are not present in the results.
        """

        if self._timepoints_averaged:
            raise ProcessingOrderError(
                "Error when processing the connectivity results: Trying to "
                "average the data across timepoints, however this has already "
                "been performed."
            )

        if "timepoints" not in self.coherence_dims:
            raise UnavailableProcessingError(
                "Error when attempting to average the timepoints in the "
                "connectivity results:\n There is no timepoints axis present "
                f"in the data. The present axes are: \n{self.coherence_dims}"
            )

        timepoints_i = self.coherence_dims.index("timepoints")

        n_timepoints = np.shape(self.coherence._data)[timepoints_i]
        self.coherence._data = np.mean(self.coherence._data, axis=timepoints_i)
        self.coherence_dims.pop(timepoints_i)

        self._timepoints_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_timepoints} timepoints.")

    def _establish_coherence_dimensions(self) -> None:
        """Establishes the dimensions of the coherence results.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if dimensions for the results are not listed for the analysis
            method used to calculate the results.
        """

        supported_modes = ["multitaper", "fourier", "cwt_morlet"]

        if self._mode in ["multitaper", "fourier"]:
            self.coherence_dims = ["channels", "frequencies"]
        elif self._mode in ["cwt_morlet"]:
            self.coherence_dims = ["channels", "frequencies", "timepoints"]
        else:
            raise UnavailableProcessingError(
                "Error when sorting the results of the connectivity analysis:\n"
                f"The analysis mode '{self._mode}' does not have an associated "
                "dimension for the results axes.\nOnly methods "
                f"'{supported_modes}' are supported."
            )

    def _sort_dimensions(self) -> None:
        """Establishes dimensions of the coherence results and averages across
        timepoints, if requested."""

        self._establish_coherence_dimensions()

        if self._average_timepoints:
            self._average_timepoints_results()

    def _generate_node_ch_names(self) -> list[list[str]]:
        """Converts the indices of channels in the connectivity results to their
        channel names.

        RETURNS
        -------
        node_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_names = [[], []]

        for group_i, indices in enumerate(self.coherence.indices):
            for index in indices:
                node_names[group_i].append(self.coherence.names[index])

        return node_names

    def _generate_node_ch_types(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the types of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_types : list[list[str]]
        -   List containing two sublists consisting of the channel types of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_ch_types = [[], []]
        ch_types = {}
        for name in self.coherence.names:
            if (
                name not in self.signal.data.ch_names
                and name[:9] == "SHUFFLED_"
            ):
                use_name = name[9:]
            else:
                use_name = name
            ch_types[name] = self.signal.data.get_channel_types(picks=use_name)[
                0
            ]

        for group_i in range(2):
            node_ch_types[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i], dict_to_order=ch_types
            )

        return node_ch_types

    def _generate_node_reref_types(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the rereferencing types of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_reref_types : list[list[str]]
        -   List containing two sublists consisting of the channel types of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_reref_types = [[], []]

        for group_i in range(2):
            node_reref_types[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i],
                dict_to_order=self.extra_info["reref_types"],
            )

        return node_reref_types

    def _generate_node_ch_coords(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the coordinates of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_coords : list[list[str]]
        -   List containing two sublists consisting of the channel coordinates
            of the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_coords = [[], []]
        ch_coords = {}
        for name in self.coherence.names:
            if (
                name not in self.signal.data.ch_names
                and name[:9] == "SHUFFLED_"
            ):
                use_name = name[9:]
            else:
                use_name = name
            ch_coords[name] = self.signal.get_coordinates()[
                self.signal.data.ch_names.index(use_name)
            ]

        for group_i in range(2):
            node_ch_coords[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i], dict_to_order=ch_coords
            )

        return node_ch_coords

    def _generate_node_ch_regions(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the regions of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_regions : list[list[str]]
        -   List containing two sublists consisting of the channel regions of
            the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_regions = [[], []]

        for group_i in range(2):
            node_ch_regions[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i],
                dict_to_order=self.extra_info["ch_regions"],
            )

        return node_ch_regions

    def _generate_node_ch_hemispheres(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the hemispheres of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_ch_hemispheres : list[list[str]]
        -   List containing two sublists consisting of the hemispheres of the
            seeds and targets, respectively, for each node in the connectivity
            results.
        """

        node_ch_hemispheres = [[], []]

        for group_i in range(2):
            node_ch_hemispheres[group_i] = ordered_list_from_dict(
                list_order=node_ch_names[group_i],
                dict_to_order=self.extra_info["ch_hemispheres"],
            )

        return node_ch_hemispheres

    def _generate_node_lateralisation(
        self, node_ch_hemispheres: list[list[str]]
    ) -> list[str]:
        """Gets the lateralisation of the channels in the connectivity node.
        -   Can either be "contralateral" if the seed and target are from
            different hemispheres, or "ipsilateral" if the seed and target are
            from the same hemisphere.

        PARAMETERS
        ----------
        node_ch_hemispheres : list[list[str]]
        -   Hemispheres of the seed and target channels of each connectivity
            node.
        -   Indication of the hemispheres should be binary in nature, e.g. "L"
            and "R", or "Left" and "Right", not "L" and "Left" and "R" and
            "Right".

        RETURNS
        -------
        node_lateralisation : list[str]
        -   Lateralisation ("contralateral" or "ipsilateral") of each
            connectivity node.
        """

        node_lateralisation = []

        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                node_lateralisation.append("contralateral")
            else:
                node_lateralisation.append("ipsilateral")

        return node_lateralisation

    def _generate_node_epoch_orders(
        self, node_ch_names: list[list[str]]
    ) -> list[list[str]]:
        """Gets the epoch orders of channels in the connectivity results.

        PARAMETERS
        ----------
        node_ch_names : list[list[str]]
        -   List containing two sublists consisting of the channel names of the
            seeds and targets, respectively, for each node in the connectivity
            results.

        RETURNS
        -------
        node_epoch_orders : list[str]
        -   List containing the epoch orders of the seeds and targets for each
            node in the connectivity results. If either the seed or target has
            a 'shuffled' epoch order, the epoch order of the node is 'shuffled',
            otherwise it is 'original'.
        """

        seed_epoch_orders = ordered_list_from_dict(
            list_order=node_ch_names[0],
            dict_to_order=self.extra_info["epoch_orders"],
        )
        target_epoch_orders = ordered_list_from_dict(
            list_order=node_ch_names[1],
            dict_to_order=self.extra_info["epoch_orders"],
        )

        node_epoch_orders = []
        for node_i in range(len(seed_epoch_orders)):
            if (
                seed_epoch_orders[node_i] == "original"
                and target_epoch_orders[node_i] == "original"
            ):
                node_epoch_orders.append("original")
            else:
                node_epoch_orders.append("shuffled")

        return node_epoch_orders

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""

        self.extra_info["node_ch_names"] = self._generate_node_ch_names()
        self.extra_info["node_ch_types"] = self._generate_node_ch_types(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info["node_reref_types"] = self._generate_node_reref_types(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info["node_ch_coords"] = self._generate_node_ch_coords(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info["node_ch_regions"] = self._generate_node_ch_regions(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info[
            "node_ch_hemispheres"
        ] = self._generate_node_ch_hemispheres(
            node_ch_names=self.extra_info["node_ch_names"]
        )
        self.extra_info[
            "node_lateralisation"
        ] = self._generate_node_lateralisation(
            node_ch_hemispheres=self.extra_info["node_ch_hemispheres"]
        )
        self.extra_info["node_epoch_orders"] = self._generate_node_epoch_orders(
            node_ch_names=self.extra_info["node_ch_names"]
        )

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""

        connectivity = spectral_connectivity_epochs(
            data=self.signal.data,
            method=self._method,
            indices=self._indices,
            sfreq=self.signal.data.info["sfreq"],
            mode=self._mode,
            fmin=self._fmin,
            fmax=self._fmax,
            fskip=self._fskip,
            faverage=self._faverage,
            tmin=self._tmin,
            tmax=self._tmax,
            mt_bandwidth=self._mt_bandwidth,
            mt_adaptive=self._mt_adaptive,
            mt_low_bias=self._mt_low_bias,
            cwt_freqs=self._cwt_freqs,
            cwt_n_cycles=self._cwt_n_cycles,
            block_size=self._block_size,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        )
        if self._shuffled_present:
            connectivity = self._sort_shuffled_results(results=connectivity)
        if self._method == "imcoh":
            connectivity._data = np.abs(connectivity._data)
        self.coherence = connectivity

        self._sort_dimensions()
        self._generate_extra_info()

    def process(
        self,
        method: str,
        mode: str,
        seeds: Optional[Union[str, list[str]]] = None,
        targets: Optional[Union[str, list[str]]] = None,
        fmin: Optional[Union[float, tuple]] = None,
        fmax: Optional[Union[float, tuple]] = float("inf"),
        fskip: int = 0,
        faverage: bool = False,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        mt_bandwidth: Optional[float] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Optional[Union[list, NDArray]] = None,
        cwt_n_cycles: Union[int, float, NDArray] = 7,
        shuffle_group: Optional[str] = None,
        n_shuffles: Optional[int] = None,
        shuffle_rng_seed: Optional[int] = None,
        average_timepoints: bool = False,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        spectral_connectivity_epochs function of the mne-connectivity package.

        PARAMETERS
        ----------
        method : str
        -   The method for calculating connectivity.
        -   Supported inputs are: 'coh' - standard coherence; 'cohy' -
            coherency; 'imcoh' - imaginary part of coherence; 'plv' -
            phase-locking value; 'ciplv' - corrected imaginary phase-locking
            value; 'ppc' - pairwise phase consistency; 'pli' - phase lag index;
            'pli2_unbiased' - unbiased estimator of squared phase lag index;
            'wpli' - weighted phase lag index; 'wpli2_debiased' - debiased
            estimator of squared weighted phase lag index.

        mode : str
        -   The mode for calculating connectivity using 'method'.
        -   Supported inputs are: 'multitaper'; 'fourier'; and 'cwt_morlet'.

        seeds : str | list[str] | None; default None
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.
        -   If None, all channels will be used as seeds.

        targets : str | list[str] | None; default None
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.
        -   If None, all channels will be used as targets.

        fmin : float | tuple | None
        -   The lower frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            lower frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their lower frequencies.
        -   If None, no lower frequency is used.

        fmax : float | tuple; default infinite
        -   The higher frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            higher frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their higher frequencies.
        -   If an infinite float, no higher frequency is used.

        fskip : int; default 0
        -   Omit every 'fskip'+1th frequency bin to decimate the frequency
            domain.
        -   If 0, no bins are skipped.

        faverage : bool; default False
        -   Whether or not to average the connectivity values for each frequency
            band.

        tmin : float | None; default None
        -   Time to start the connectivity estimation.
        -   If None, the data is used from the beginning.

        tmax : float | None; default None
        -   Time to end the connectivity estimation.
        -   If None, the data is used until the end.

        mt_bandwidth : float | None
        -   The bandwidth, in Hz, of the multitaper windowing function.
        -   Only used if 'mode' is 'multitaper'.

        mt_adaptive : bool; default False
        -   Whether or not to use adaptive weights to comine the tapered spectra
            into power spectra.
        -   Only used if 'mode' is 'multitaper'.

        mt_low_bias : bool: default True
        -   Whether or not to only use tapers with > 90% spectral concentration
            within bandwidth.
        -   Only used if 'mode' is 'multitaper'.

        cwt_freqs: list[int | float] | array[int | float] | None
        -   The frequencies of interest to calculate connectivity for.
        -   Only used if 'mode' is 'cwt_morlet'. In this case, 'cwt_freqs'
            cannot be None.

        cwt_n_cycles: int | float | array[int | float]; default 7
        -   The number of cycles to use when calculating connectivity.
        -   If an single integer or float, this number of cycles is for each
            frequency.
        -   If an array, the entries correspond to the number of cycles to use
            for each frequency being analysed.
        -   Only used if 'mode' is 'cwt_morlet'.

        shuffle_group: str | None
        -   The group of channels to create epoch-shuffled data for. Supported
            groups are 'seeds' or 'targets', corresponding to the channels in
            those groups.
        -   If None, no shuffled data is created or analysed.

        n_shuffles : int | None
        -   How many times to create epoch-shuffled data for each channel in
            'shuffle_group', whose connectivity results are then averaged over.
        -   Only used if 'shuffle_group' is not None. In this case, 'n_shuffles'
            cannot be None.

        shuffle_rng_sheed : int | None
        -   The seed to use for the random number generator for shuffling the
            epoch order.
        -   If None, no seed is used.
        -   Only used if 'shuffle_group' is not None.

        block_size : int; default 1000
        -   The number of connections to compute at once.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.
        """

        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self._method = method
        self._mode = mode
        self._seeds = seeds
        self._targets = targets
        self._fmin = fmin
        self._fmax = fmax
        self._fskip = fskip
        self._faverage = faverage
        self._tmin = tmin
        self._tmax = tmax
        self._mt_bandwidth = mt_bandwidth
        self._mt_adaptive = mt_adaptive
        self._mt_low_bias = mt_low_bias
        self._cwt_freqs = cwt_freqs
        self._cwt_n_cycles = cwt_n_cycles
        self._shuffle_group = shuffle_group
        self._n_shuffles = n_shuffles
        self._shuffle_rng_seed = shuffle_rng_seed
        self._average_timepoints = average_timepoints
        self._block_size = block_size
        self._n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

        self._processed = True
        self.processing_steps["connectivity_coherence"] = {
            "method": method,
            "mode": mode,
            "seeds": seeds,
            "targets": targets,
            "fmin": fmin,
            "fmax": fmax,
            "fskip": fskip,
            "faverage": faverage,
            "tmin": tmin,
            "tmax": tmax,
            "mt_bandwidth": mt_bandwidth,
            "mt_adaptive": mt_adaptive,
            "mt_low_bias": mt_low_bias,
            "cwt_freqs": cwt_freqs,
            "cwt_n_cycles": cwt_n_cycles,
            "shuffle_group": shuffle_group,
            "n_shuffles": n_shuffles,
            "shuffle_rng_seed": shuffle_rng_seed,
            "average_timepoints": average_timepoints,
        }

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the ConnectivityCoherence object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   Whether or not the user is asked to confirm to overwrite a
            pre-existing file if one exists.
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_object(
            to_save=self,
            fpath=fpath,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def save_results(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the coherence results and additional information as a file.

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
        """

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_dict(
            to_save=self.results_as_dict(),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def results_as_dict(self) -> dict:
        """Returns the coherence results and additional information as a
        dictionary.

        RETURNS
        -------
        dict
        -   The results and additional information stored as a dictionary.
        """

        return {
            f"connectivity-{self._method}": self.coherence._data.tolist(),
            f"connectivity-{self._method}_dimensions": self.coherence_dims,
            "freqs": self.coherence.freqs,
            "seed_names": self.extra_info["node_ch_names"][0],
            "seed_types": self.extra_info["node_ch_types"][0],
            "seed_coords": self.extra_info["node_ch_coords"][0],
            "seed_regions": self.extra_info["node_ch_regions"][0],
            "seed_hemispheres": self.extra_info["node_ch_hemispheres"][0],
            "seed_reref_types": self.extra_info["node_reref_types"][0],
            "target_names": self.extra_info["node_ch_names"][1],
            "target_types": self.extra_info["node_ch_types"][1],
            "target_coords": self.extra_info["node_ch_coords"][1],
            "target_regions": self.extra_info["node_ch_regions"][1],
            "target_hemispheres": self.extra_info["node_ch_hemispheres"][1],
            "target_reref_types": self.extra_info["node_reref_types"][1],
            "node_lateralisation": self.extra_info["node_lateralisation"],
            "node_epoch_orders": self.extra_info["node_epoch_orders"],
            "samp_freq": self.signal.data.info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data.info["subject_info"],
        }
