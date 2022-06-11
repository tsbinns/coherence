"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from mne_connectivity import seed_target_indices
from coh_handle_entries import (
    combine_vals_list,
    get_eligible_idcs_lists,
    get_group_names_idcs,
    ordered_list_from_dict,
    unique,
)
import coh_signal


class ProcMethod(ABC):
    """Abstract class for implementing data processing methods.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   A preprocessed Signal object whose data will be processed.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process (abstract)
    -   Processes the data.

    save_object (abstract)
    -   Saves the object as a .pkl file.

    save_results (abstract)
    -   Converts the results and additional information to a dictionary and
        saves them as a file.

    results_as_dict (abstract)
    -   Organises the results and additional information into a dictionary.
    """

    @abstractmethod
    def __init__(self, signal: coh_signal.Signal, verbose: bool) -> None:

        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self.results = None
        self._results_dims = None
        self.processing_steps = None
        self.extra_info = None

        # Initialises inputs of the ProcMethod object.
        self.signal = deepcopy(signal)
        self._verbose = verbose

        # Initialises aspects of the ProcMethod object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._processed = False
        self._windows_averaged = False

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

    @abstractmethod
    def _get_results(self) -> None:
        """Performs the analysis to get the results."""

    @abstractmethod
    def _sort_inputs(self) -> None:
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing and assigns inputs."""

        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.extra_info = deepcopy(self.signal.extra_info)

    @property
    def results_dims(self) -> list[str]:
        """Returns the dimensions of the results, corresponding to the results
        that will be returned with the 'get_results' method.

        RETURNS
        -------
        dims : list[str]
        -   Dimensions of the results.
        """

        if self._windows_averaged:
            dims = self._results_dims[1:]
        else:
            dims = self._results_dims

        return deepcopy(dims)

    @abstractmethod
    def save_object(self) -> None:
        """Saves the object as a .pkl file."""

    @abstractmethod
    def save_results(self) -> None:
        """Converts the results and additional information to a dictionary and
        saves them as a file."""

    @abstractmethod
    def results_as_dict(self) -> None:
        """Organises the results and additional information into a
        dictionary."""

    def _get_optimal_dims(self) -> list[str]:
        """Finds the optimal order of dimensions for the results, following the
        order ["windows", "channels", "epochs", "frequencies", "timepoints"]
        based on which dimensions are present in the reuslts.

        RETURNS
        -------
        optimal_dims : list[str]
        -   Optimal dimensions of the results.
        """

        possible_order = [
            "windows",
            "channels",
            "epochs",
            "frequencies",
            "timepoints",
        ]
        optimal_dims = [
            dim for dim in possible_order if dim in self.results_dims
        ]

        return optimal_dims


class ProcConnectivity(ProcMethod):
    """Class for processing connectivity results. A subclass of 'ProcMethod'.

    METHODS
    -------
    process (abstract)
    -   Processes the data.

    save_object (abstract)
    -   Saves the object as a .pkl file.

    save_results (abstract)
    -   Converts the results and additional information to a dictionary and
        saves them as a file.

    results_as_dict (abstract)
    -   Organises the results and additional information into a dictionary.
    """

    @abstractmethod
    def __init__(self, signal: coh_signal.Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)

        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self._indices = None
        self._seeds_list = None
        self._targets_list = None
        self._seeds_str = None
        self._targets_str = None
        self._node_ch_names = None

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

    @abstractmethod
    def _get_results(self) -> None:
        """Performs the analysis to get the results."""

    def _sort_inputs(self) -> None:
        """Checks the inputs to the object to ensure that they match the
        requirements for processing and assigns inputs.

        RAISES
        ------
        ValueError
        -   Raised if the dimensions of the data in the Signal object is not
            supported.
        """

        supported_data_dims = ["windows", "epochs", "channels", "timepoints"]
        if self.signal.data_dimensions != supported_data_dims:
            raise ValueError(
                "Error when trying to perform coherence analysis on the "
                "data:\nData in the Signal object has the dimensions "
                f"{self.signal.data_dimensions}, but only data with dimensions "
                f"{supported_data_dims} is supported."
            )

        super()._sort_inputs()

    def _sort_seeds_targets(self) -> None:
        """Sorts the names of the seeds and targets for the connectivity
        analysis, and generates the corresponding channel indices.

        If the seeds and/or targets are dictionaries, the names of the seeds and
        targets will be automatically generated based on the information in the
        dictionaries, and then expanded, such that connectivity is calculated
        between every seed and every target.

        If the seeds and targets are both lists, the channel names in these
        lists are taken as the seeds and targets and no expansion is performed.
        """

        groups = ["_seeds", "_targets"]
        features = self._features_to_df()
        groups_vals = [getattr(self, group) for group in groups]
        expand_seeds_targets = False
        for group_i, group in enumerate(groups):
            group_vals = groups_vals[group_i]
            if isinstance(group_vals, dict):
                expand_seeds_targets = True
                eligible_idcs = get_eligible_idcs_lists(
                    features, group_vals["eligible_entries"]
                )
                group_idcs = get_group_names_idcs(
                    features,
                    group_vals["grouping"],
                    eligible_idcs=eligible_idcs,
                    replacement_idcs=eligible_idcs,
                )
                names_list = []
                names_str = []
                for idcs in group_idcs.values():
                    names_list.append(
                        [self.signal.data[0].ch_names[idx] for idx in idcs]
                    )
                    names_str.append(
                        combine_vals_list(
                            [self.signal.data[0].ch_names[idx] for idx in idcs]
                        )
                    )
                setattr(self, f"{group}_list", names_list)
                setattr(self, f"{group}_str", names_str)

        if expand_seeds_targets:
            self._expand_seeds_targets()

        if len(self._seeds_list) != len(self._targets_list):
            raise ValueError(
                "Seeds and targets must contain the same number of entries, "
                f"but do not ({len(self._seeds_list)} and "
                f"{len(self._targets_list)}, respectively)."
            )

        self._get_names_indices_mne()

    def _features_to_df(self) -> pd.DataFrame:
        """Collates features of channels (e.g. names, types, regions, etc...)
        into a pandas DataFrame so that which channels belong to which groups
        can be easily checked.

        RETURNS
        -------
        pandas DataFrame
        -   DataFrame containing the features of each channel.
        """
        ch_names = self.signal.data[0].ch_names
        features = {
            "ch_names": ch_names,
            "ch_types": self.signal.data[0].get_channel_types(picks=ch_names),
            "ch_regions": ordered_list_from_dict(
                ch_names, self.extra_info["ch_regions"]
            ),
            "ch_hemispheres": ordered_list_from_dict(
                ch_names, self.extra_info["ch_hemispheres"]
            ),
            "ch_reref_types": ordered_list_from_dict(
                ch_names, self.extra_info["ch_reref_types"]
            ),
            "ch_epoch_orders": ordered_list_from_dict(
                ch_names, self.extra_info["ch_epoch_orders"]
            ),
        }

        return pd.DataFrame(features)

    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group.
        -   Should be used when seeds and/or targets have been automatically
            generated based on channel types."""

        seeds_list = []
        targets_list = []
        seeds_str = []
        targets_str = []
        for seed in self._seeds_list:
            for target in self._targets_list:
                seeds_list.append(seed)
                targets_list.append(target)
                seeds_str.append(combine_vals_list(seed))
                targets_str.append(combine_vals_list(target))

        self._seeds_list = seeds_list
        self._targets_list = targets_list
        self._seeds_str = seeds_str
        self._targets_str = targets_str

    def _get_names_indices_mne(self) -> None:
        """Gets the names and indices of seed and targets in the connectivity
        analysis for use in an MNE connectivity object.

        As MNE connectivity objects only support seed-target pair names and
        indices between two channels, the names of channels in each group of
        seeds and targets are combined together, and the indices then derived
        from these combined names.
        """
        seed_names_str = []
        target_names_str = []
        for seeds, targets in zip(self._seeds_list, self._targets_list):
            seed_names_str.append(combine_vals_list(seeds))
            target_names_str.append(combine_vals_list(targets))
        unique_names_str = [*unique(seed_names_str), *unique(target_names_str)]
        unique_names_list = [
            *unique(self._seeds_list),
            *unique(self._targets_list),
        ]
        indices = [[], []]
        for seeds, targets in zip(seed_names_str, target_names_str):
            indices[0].append(unique_names_str.index(seeds))
            indices[1].append(unique_names_str.index(targets))

        self._comb_names_str = unique_names_str
        self._comb_names_list = unique_names_list
        self._indices = np.asarray(indices)

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""

        self._indices = seed_target_indices(
            seeds=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self._seeds_list
            ],
            targets=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self._targets_list
            ],
        )

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""

        self._generate_node_ch_names()
        self._generate_node_ch_types()
        self._generate_node_ch_reref_types()
        self._generate_node_ch_coords()
        self._generate_node_ch_regions()
        self._generate_node_ch_hemispheres()
        self._generate_node_lateralisation()
        self._generate_node_ch_epoch_orders()

    def _generate_node_ch_names(self) -> None:
        """Converts the indices of channels in the connectivity results to their
        channel names.
        -   Names are a list containing two sublists consisting of the channel
            names of the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_names = [[], []]
        for group_i, indices in enumerate(self.results[0].indices):
            for index in indices:
                node_ch_names[group_i].append(self.results[0].names[index])
        self.extra_info["node_ch_names"] = node_ch_names

    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results.
        -   Types are a list containing two sublists consisting of the channel
            types of the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_types = [[], []]
        ch_types = {}
        for name in self.results[0].names:
            ch_types[name] = self.signal.data[0].get_channel_types(picks=name)[
                0
            ]

        for group_i in range(2):
            node_ch_types[group_i] = ordered_list_from_dict(
                list_order=self.extra_info["node_ch_names"][group_i],
                dict_to_order=ch_types,
            )
        self.extra_info["node_ch_types"] = node_ch_types

    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity results.
        -   Rereferencing types are a list containing two sublists consisting of
            the rereferencing types of the seeds and targets, respectively, for
            each node in the connectivity results.
        """

        node_ch_reref_types = [[], []]
        for group_i in range(2):
            node_ch_reref_types[group_i] = ordered_list_from_dict(
                list_order=self.extra_info["node_ch_names"][group_i],
                dict_to_order=self.extra_info["ch_reref_types"],
            )
        self.extra_info["node_ch_reref_types"] = node_ch_reref_types

    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results.
        -   Coordinates are a list containing two sublists consisting of the
            channel coordinates of the seeds and targets, respectively, for each
            node in the connectivity results.
        """

        node_ch_coords = [[], []]
        ch_coords = {
            name: self.signal.get_coordinates(name)[0]
            for name in self.results[0].names
        }
        for group_i in range(2):
            node_ch_coords[group_i] = ordered_list_from_dict(
                list_order=self.extra_info["node_ch_names"][group_i],
                dict_to_order=ch_coords,
            )
        self.extra_info["node_ch_coords"] = node_ch_coords

    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results.
        -   Regions are lists containing two sublists consisting of the channel
            regions of the seeds and targets, respectively, for each node in the
            connectivity results.
        """

        node_ch_regions = [[], []]
        for group_i in range(2):
            node_ch_regions[group_i] = ordered_list_from_dict(
                list_order=self.extra_info["node_ch_names"][group_i],
                dict_to_order=self.extra_info["ch_regions"],
            )
        self.extra_info["node_ch_regions"] = node_ch_regions

    def _generate_node_ch_hemispheres(self) -> None:
        """Gets the hemispheres of channels in the connectivity results.
        -   Hemispheres are lists containing two sublists consisting of the
            hemispheres of the seeds and targets, respectively, for each node in
            the connectivity results.
        """

        node_ch_hemispheres = [[], []]
        for group_i in range(2):
            node_ch_hemispheres[group_i] = ordered_list_from_dict(
                list_order=self.extra_info["node_ch_names"][group_i],
                dict_to_order=self.extra_info["ch_hemispheres"],
            )
        self.extra_info["node_ch_hemispheres"] = node_ch_hemispheres

    def _generate_node_lateralisation(self) -> None:
        """Gets the lateralisation of the channels in the connectivity node.
        -   Can either be "contralateral" if the seed and target are from
            different hemispheres, or "ipsilateral" if the seed and target are
            from the same hemisphere.
        -   Indication of the hemispheres should be binary in nature, e.g. "L"
            and "R", or "Left" and "Right", not "L" and "Left" and "R" and
            "Right".
        -   Lateralisation is a list of strings with values "contralateral" or
            "ipsilateral" for each connectivity node.
        """

        node_lateralisation = []
        node_ch_hemispheres = self.extra_info["node_ch_hemispheres"]
        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                node_lateralisation.append("contralateral")
            else:
                node_lateralisation.append("ipsilateral")

        self.extra_info["node_lateralisation"] = node_lateralisation

    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results.
        -   Epoch orders are a list containing the epoch orders of the seeds and
            targets for each node in the connectivity results. If either the
            seed or target has a 'shuffled' epoch order, the epoch order of the
            node is 'shuffled', otherwise it is 'original'.
        """

        seed_epoch_orders = ordered_list_from_dict(
            list_order=self.extra_info["node_ch_names"][0],
            dict_to_order=self.extra_info["ch_epoch_orders"],
        )
        target_epoch_orders = ordered_list_from_dict(
            list_order=self.extra_info["node_ch_names"][1],
            dict_to_order=self.extra_info["ch_epoch_orders"],
        )
        node_ch_epoch_orders = []
        for seed_order, target_order in zip(
            seed_epoch_orders, target_epoch_orders
        ):
            if seed_order == "original" and target_order == "original":
                node_ch_epoch_orders.append("original")
            else:
                node_ch_epoch_orders.append("shuffled")
        self.extra_info["node_ch_epoch_orders"] = node_ch_epoch_orders

    @abstractmethod
    def save_object(self) -> None:
        """Saves the object as a .pkl file."""

    @abstractmethod
    def save_results(self) -> None:
        """Converts the results and additional information to a dictionary and
        saves them as a file."""

    @abstractmethod
    def results_as_dict(self) -> None:
        """Organises the results and additional information into a
        dictionary."""
