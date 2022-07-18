"""Abstract subclasses for implementing connectivity processing methods.

CLASSES
-------
ProcConnectivity
-   Class for processing connectivity results. A subclass of 'ProcMethod'.

ProcSingularConnectivity
-   Class for processing connectivity results between pairs of single channels.
    A subclass of 'ProcConnectivity'.

ProcMultivariateConnectivity
-   Class for processing multivariate connectivity results. A subclass of
    'ProcConnectivity'.
"""

from abc import abstractmethod
from copy import deepcopy
from typing import Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from mne_connectivity import seed_target_indices, SpectralConnectivity
from coh_exceptions import ProcessingOrderError
from coh_handle_entries import (
    combine_vals_list,
    get_eligible_idcs_lists,
    get_group_names_idcs,
    ordered_list_from_dict,
    rearrange_axes,
    unique,
)
from coh_processing_methods import ProcMethod
from coh_signal import Signal


class ProcConnectivity(ProcMethod):
    """Class for processing connectivity results. A subclass of 'ProcMethod'.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool
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
    def __init__(self, signal: Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

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

    @abstractmethod
    def _sort_seeds_targets(self) -> None:
        """Sorts the names of the seeds and targets for the connectivity
        analysis, and generates the corresponding channel indices."""

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
        return pd.DataFrame(
            {
                "ch_names": ch_names,
                "ch_types": self.signal.data[0].get_channel_types(
                    picks=ch_names
                ),
                "ch_regions": ordered_list_from_dict(
                    ch_names, self.extra_info["ch_regions"]
                ),
                "ch_subregions": ordered_list_from_dict(
                    ch_names, self.extra_info["ch_subregions"]
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
        )

    @abstractmethod
    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group."""

    @abstractmethod
    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""

    @abstractmethod
    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity
        results."""

    @abstractmethod
    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results,
        averaged across for each channel in the seeds and targets."""

    @abstractmethod
    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_ch_subregions(self) -> None:
        """Gets the subregions of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_ch_hemispheres(self) -> None:
        """Gets the hemispheres of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_lateralisation(self) -> None:
        """Gets the lateralisation of the channels in the connectivity node."""

    @abstractmethod
    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results."""

    @abstractmethod
    def save_results(self) -> None:
        """Converts the results and additional information to a dictionary and
        saves them as a file."""

    @abstractmethod
    def results_as_dict(self) -> None:
        """Organises the results and additional information into a
        dictionary."""

    def get_results(self, dimensions: Union[list[str], None] = None) -> NDArray:
        """Extracts and returns results.

        PARAMETERS
        ----------
        dimensions : list[str] | None;  default None
        -   The dimensions of the results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        results : numpy array
        -   The results.
        """

        if dimensions is None:
            dimensions = self.results_dims

        if self._windows_averaged:
            results = self.results[0].get_data()
        else:
            results = []
            for mne_obj in self.results:
                results.append(mne_obj.get_data())
            results = np.asarray(results)

        results = rearrange_axes(
            obj=results, old_order=self.results_dims, new_order=dimensions
        )

        return deepcopy(results)


class ProcSingularConnectivity(ProcConnectivity):
    """Class for processing connectivity results between pairs of single
    channels. A subclass of 'ProcConnectivity'.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool
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
    def __init__(self, signal: Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)
        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self._indices = None
        self._seeds = None
        self._targets = None

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

    def _sort_seeds_targets(self) -> None:
        """Sorts the names of the seeds and targets for the connectivity
        analysis, and generates the corresponding channel indices.

        If the seeds and/or targets are dictionaries, the names of the seeds and
        targets will be automatically generated based on the information in the
        dictionaries, and then expanded, such that connectivity is calculated
        between every seed and every target.

        If the seeds and targets are both lists, the channel names in these
        lists are taken as the seeds and targets and no expansion is performed.

        RAISES
        ------
        ValueError
        -   Raised if the seeds and/or targets contain multiple channels per
            node.
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
                names = []
                for idx in group_idcs.values():
                    if len(idx) > 1:
                        raise ValueError(
                            "For singular connectivity, seeds and targets for "
                            "a node can only contain one channel each, however "
                            f"a node of the {group[1:]} contains {len(idx)} "
                            "channels.\nIf you wish to compute connectivity "
                            "between groups of channels, please use one of the "
                            "multivariate connectivity methods."
                        )
                    names.append(self.signal.data[0].ch_names[idx[0]])
                setattr(self, f"{group}", names)
            elif isinstance(group_vals, list):
                names = []
                for val in group_vals:
                    if not isinstance(val, str):
                        raise TypeError(
                            "Seeds and targets must be specified as strings, "
                            "as for singular connectivity, seeds and targets "
                            "can only contain one channel each.\nIf you wish "
                            "to compute connectivity between groups of "
                            "channels, please use one of the multivariate "
                            "connectivity methods."
                        )
                    names.append(val)
                setattr(self, f"{group}", names)
            else:
                raise TypeError(
                    "Seeds and targets must given as lists, or as dictionaries "
                    "with instructions for generating these lists, however the "
                    f"{group[1:]} are of type {type(group_vals)}."
                )

        if expand_seeds_targets:
            self._expand_seeds_targets()

        if len(self._seeds) != len(self._targets):
            raise ValueError(
                "Seeds and targets must contain the same number of entries, "
                f"but do not ({len(self._seeds)} and {len(self._targets)}, "
                "respectively)."
            )

        self._generate_indices()

    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group.

        Should be used when seeds and/or targets have been automatically
        generated based on channel types.
        """
        seeds = []
        targets = []
        for seed in self._seeds:
            for target in self._targets:
                seeds.append(seed)
                targets.append(target)
        self._seeds = seeds
        self._targets = targets

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""
        self._indices = seed_target_indices(
            seeds=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self._seeds
            ],
            targets=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self._targets
            ],
        )

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""
        self._generate_node_ch_types()
        self._generate_node_ch_reref_types()
        self._generate_node_ch_coords()
        self._generate_node_ch_regions()
        self._generate_node_ch_subregions()
        self._generate_node_ch_hemispheres()
        self._generate_node_lateralisation()
        self._generate_node_ch_epoch_orders()

    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results."""
        node_ch_types = [[], []]
        groups = ["_seeds", "_targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_types[group_i].append(
                    self.signal.data[0].get_channel_types(picks=name)[0]
                )
        self.extra_info["node_ch_types"] = node_ch_types

    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity
        results."""
        node_reref_types = [[], []]
        groups = ["_seeds", "_targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_reref_types[group_i].append(
                    self.extra_info["ch_reref_types"][name]
                )
        self.extra_info["node_ch_reref_types"] = node_reref_types

    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results,
        averaged across for each channel in the seeds and targets."""
        node_ch_coords = [[], []]
        groups = ["_seeds", "_targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_coords[group_i].append(
                    self.signal.get_coordinates(name)[0]
                )
        self.extra_info["node_ch_coords"] = node_ch_coords

    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results."""
        node_ch_regions = [[], []]
        groups = ["_seeds", "_targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_regions[group_i].append(
                    self.extra_info["ch_regions"][name]
                )
        self.extra_info["node_ch_regions"] = node_ch_regions

    def _generate_node_ch_subregions(self) -> None:
        """Gets the subregions of channels in the connectivity results."""
        node_ch_subregions = [[], []]
        groups = ["_seeds", "_targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_subregions[group_i].append(
                    self.extra_info["ch_subregions"][name]
                )
        self.extra_info["node_ch_subregions"] = node_ch_subregions

    def _generate_node_ch_hemispheres(self) -> None:
        """Gets the hemispheres of channels in the connectivity results."""
        node_ch_hemispheres = [[], []]
        groups = ["_seeds", "_targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_hemispheres[group_i].append(
                    self.extra_info["ch_hemispheres"][name]
                )
        self.extra_info["node_ch_hemispheres"] = node_ch_hemispheres

    def _generate_node_lateralisation(self) -> None:
        """Gets the lateralisation of the channels in the connectivity node.

        Can either be "contralateral" if the seed and target are from different
        hemispheres, or "ipsilateral" if the seed and target are from the same
        hemisphere.
        """
        node_lateralisation = []
        node_ch_hemispheres = self.extra_info["node_ch_hemispheres"]
        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                lateralisation = "contralateral"
            else:
                lateralisation = "ipsilateral"
            node_lateralisation.append(lateralisation)
        self.extra_info["node_lateralisation"] = node_lateralisation

    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results.

        If either the seed or target has a "shuffled" epoch order, the epoch
        order of the node is "shuffled", otherwise it is "original".
        """
        node_epoch_orders = []
        for seed_name, target_name in zip(self._seeds, self._targets):
            if (
                self.extra_info["ch_epoch_orders"][seed_name] == "original"
                and self.extra_info["ch_epoch_orders"][target_name]
                == "original"
            ):
                order = "original"
            else:
                order = "shuffled"
            node_epoch_orders.append(order)
        self.extra_info["node_ch_epoch_orders"] = node_epoch_orders

    @abstractmethod
    def save_results(self) -> None:
        """Converts the results and additional information to a dictionary and
        saves them as a file."""

    @abstractmethod
    def results_as_dict(self) -> None:
        """Organises the results and additional information into a
        dictionary."""


class ProcMultivariateConnectivity(ProcConnectivity):
    """Class for processing multivariate connectivity results. A subclass of
    'ProcConnectivity'.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool
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
    def __init__(self, signal: Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)
        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self._indices = None
        self._seeds_list = None
        self._targets_list = None
        self._seeds_str = None
        self._targets_str = None
        self._seed_ranks = None
        self._target_ranks = None
        self._comb_names_str = None
        self._comb_names_list = None
        self._ensure_full_rank_data = None

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

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
            elif isinstance(group_vals, list):
                names_str = [combine_vals_list(val for val in group_vals)]
                setattr(self, f"{group}_str", names_str)
                setattr(self, f"{group}_list", group_vals)
            else:
                raise TypeError(
                    "Seeds and targets must given as lists, or as dictionaries "
                    "with instructions for generating these lists, however the "
                    f"{group[1:]} are of type {type(group_vals)}."
                )

        if expand_seeds_targets:
            self._expand_seeds_targets()

        if len(self._seeds_list) != len(self._targets_list):
            raise ValueError(
                "Seeds and targets must contain the same number of entries, "
                f"but do not ({len(self._seeds_list)} and "
                f"{len(self._targets_list)}, respectively)."
            )

        self._get_names_indices_mne()

    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group.

        Should be used when seeds and/or targets have been automatically
        generated based on channel types.
        """
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
        self._indices = tuple(indices)

    def _extract_data(self, names: list[str]) -> NDArray:
        """Gets the data for a set of channels.

        PARAMETERS
        ----------
        names : list[str]
        -   Names of the channels whose data should be returned.

        RETURNS
        -------
        numpy ndarray
        -   A 4D matrix of the channels' data with dimensions [windows x epochs
            x channels x timepoints].
        """
        data = []
        for win_data in self.signal.data:
            data.append(win_data.get_data(picks=names))
        return np.asarray(data)

    def _check_data_rank(
        self, seed_data: NDArray, target_data: NDArray
    ) -> tuple[NDArray, int, NDArray, int]:
        """Checks whether the seed and target data for a node has full rank,
        performing a singular value decomposition (SVD) on the data if not and
        returning only the number of components equal to the number of non-zero
        singular values of the data.

        PARAMETERS
        ----------
        seed_data : numpy ndarray
        -   A 4D matrix of the seed data with dimensions [windows x epochs x
            channels x timepoints].

        target_data : numpy ndarray
        -   A 4D matrix of the target data with dimensions [windows x epochs x
            channels x timepoints].

        RETURNS
        -------
        seed_data : numpy ndarray
        -   A 4D matrix of the seed data with dimensions [windows x epochs x
            rank x timepoints]. If the data has full rank, the data is not
            altered, else data with full rank is returned.

        target_data : numpy ndarray
        -   A 4D matrix of the target data with dimensions [windows x epochs x
            rank x timepoints]. If the data has full rank, the data is not
            altered, else data with full rank is returned.

        seed_rank : int
        -   The rank of the seed data.

        target_rank : int
        -   The rank of the target data.
        """
        seed_data, seed_rank = self._sort_data_dimensionality(
            data=seed_data, data_type="seed"
        )
        target_data, target_rank = self._sort_data_dimensionality(
            data=target_data, data_type="target"
        )
        self._seed_ranks.append(seed_rank)
        self._target_ranks.append(target_rank)

        return seed_data, target_data, seed_rank, target_rank

    def _sort_data_dimensionality(
        self, data: NDArray, data_type: str = ""
    ) -> NDArray:
        """Ensures that the data is full rank, performing a singular value
        decomposition (SVD) and taking only the number of components with
        non-zero singular values if the data lacks full rank.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   A 4D matrix of the data with dimensions [windows x epochs x channels
            x timepoints].

        data_type : str; default ""
        -   The name of the data type being processed, e.g. "seed", "target",
            etc..., for use in raising warnings about non-full rank data.

        RETURNS
        -------
        sorted_data : numpy ndarray
        -   A 4D matrix of the data with dimensions [windows x epochs x rank x
            timepoints]. If the data has full rank, no SVD is performed and the
            original data is returned.

        data_rank : int
        -   The rank of the data.
        """
        data_combined, data_rank, data_dims = self._check_data_dimensionality(
            data
        )
        if data_rank != data_combined.shape[0]:
            data_combined, _, _ = self._get_full_rank_data(data_combined)
            if self._verbose:
                print(
                    f"The {data_type} data lacks full rank (rank {data_rank} "
                    f"of {data.shape[2]} components). Taking only those "
                    f"{data_rank} components with non-zero singular values.\n"
                )
        data_dims = [data_rank, *data_dims[1:]]
        sorted_data = self._restore_data_dimensions(
            data=data_combined, dimensions=data_dims
        )

        return sorted_data, data_rank

    def _check_data_dimensionality(
        self, data: NDArray
    ) -> tuple[NDArray, int, list[int]]:
        """Converts windowed and epoched data into standard timeseries data and
        checks whether the data has full rank.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   A 4D matrix of the data with dimensions [windows x epochs x channels
            x timepoints].

        RETURNS
        -------
        data_combined : numpy ndarray
        -   A 2D matrix of the data with dimensions [channels x timepoints].

        data_rank : int
        -   The rank of the data.

        data_dims : list[int]
        -   The dimensions of the data prior to being recombined into standard
            timeseries data, consisting of [channels, windows, epochs,
            timepoints].
        """
        data_combined, data_dims = self._recombine_data(data)
        data_rank = np.linalg.matrix_rank(data_combined)

        return data_combined, data_rank, data_dims

    def _recombine_data(self, data: NDArray) -> tuple[NDArray, list[int]]:
        """Recombines windowed and epoched data into a 2D matrix with dimensions
        [channels x timepoints].

        PARAMETERS
        ----------
        data : numpy ndarray
        -   A 4D matrix with dimensions [windows x epochs x channels x
            timepoints].

        RETURNS
        -------
        numpy ndarray
        -   The data recombined over windows and epochs with dimensions
            [channels x timepoints].

        list[int]
        -   The dimensions of the data prior to being recombined, consisting of
            [channels, windows, epochs, timepoints].
        """
        n_windows, n_epochs, n_channels, n_timepoints = data.shape
        recombined_data = data.transpose((2, 0, 1, 3))
        recombined_data = recombined_data.reshape(
            n_channels, n_windows * n_epochs * n_timepoints
        )
        return recombined_data, [n_channels, n_windows, n_epochs, n_timepoints]

    def _get_full_rank_data(
        self, data: NDArray
    ) -> tuple[NDArray, int, NDArray]:
        """Performs a single value decomposition (SVD) on the data and takes
        only the number of components equal to the number of non-zero singular
        values (i.e. the rank of the matrix), ensuring that the returned data
        has full rank.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   Data with dimensions [channels x timepoints].

        RETURNS
        -------
        V : numpy ndarray
        -   Full rank data with dimensions [rank x timepoints]. Derived from V
            of the SVD.

        rank : int
        -   The rank of the data.

        U : numpy array
        -   Data with dimensions [channels x rank]. Derived from U of the SVD.
        """
        rank = np.linalg.matrix_rank(data)
        U, _, V = np.linalg.svd(data, full_matrices=False)
        return V[:rank, :], rank, U[:, :rank]

    def _restore_data_dimensions(
        self, data: NDArray, dimensions: list[int]
    ) -> NDArray:
        """Converts the recombined data back into its windowed and epoched form.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   A 2D matrix with dimensions [channels x timepoints].

        dimensions : list[int]
        -   The shape of the data in its form prior to being recombined,
            consisting of [channels, windows, epochs, timepoints].

        RETURNS
        -------
        numpy ndarray
        -   The data in its original form as a 4D matrix with dimensions
            [windows x epochs x channels x timepoints].
        """
        restored_data = data.reshape(dimensions)
        return restored_data.transpose(1, 2, 0, 3)

    def _join_seed_target_data(
        self, seed_data: NDArray, target_data: NDArray
    ) -> list[NDArray]:
        """Rejoins seed and target data into a single set of matrices.

        PARAMETERS
        ----------
        seed_data : numpy ndarray
        -   The seed data in a 4D matrix with dimensions [windows x epochs x
            channels x timepoints].

        target_data : numpy ndarray
        -   The target data in a 4D matrix with dimensions [windows x epochs x
            channels x timepoints].

        RETURNS
        -------
        data : list[numpy ndarray]
        -   The seed and target data in a set of 3D matrices with dimensions
            [epochs x channels x timepoints], stored in a list corresponding to
            the data of different windows.
        """
        data = []
        joined_data = np.concatenate((seed_data, target_data), 2)
        for win_i in range(joined_data.shape[0]):
            data.append(joined_data[win_i, :, :, :])
        return data

    def _multivariate_to_mne(
        self,
        data: NDArray,
        freqs: NDArray,
        n_epochs_used: int,
        connectivity_method=str,
    ) -> SpectralConnectivity:
        """Converts results of the multivariate connectivity analysis stored as
        a numpy array into an MNE SpectralConnectivity object.

        PARAMETERS
        ----------
        data : numpy array
        -   Results of the multivariate connectivity analysis with dimensions
            [nodes x frequencies].

        freqs : numpy array
        -   The frequencies in the connectivity data.

        n_epochs_used : int
        -   The number of epochs used to compute connectivity.

        connectivity_method : str
        -   The name of the method used to compute the connectivity.

        RETURNS
        -------
        MNE SpectralConnectivity
        -   Results converted to an appropriate MNE object.
        """
        return SpectralConnectivity(
            data=data,
            freqs=freqs,
            n_nodes=len(self._seeds_list),
            names=self._comb_names_str,
            indices=self._indices,
            method=connectivity_method,
            n_epochs_used=n_epochs_used,
        )

    def _average_windows_results(self) -> None:
        """Averages the connectivity results across windows.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the windows have already been averaged across.
        """
        if self._windows_averaged:
            raise ProcessingOrderError(
                "Error when averaging the connectivity results across "
                "windows:\nResults have already been averaged across windows."
            )

        n_windows = len(self.results)
        self.results = [
            SpectralConnectivity(
                data=np.asarray(
                    [data.get_data() for data in self.results]
                ).mean(axis=0),
                freqs=self.results[0].freqs,
                n_nodes=self.results[0].n_nodes,
                names=self.results[0].names,
                indices=self.results[0].indices,
                method=self.results[0].method,
                n_epochs_used=self.results[0].n_epochs_used,
            )
        ]

        self._windows_averaged = True
        if self._verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""
        self._generate_node_ch_types()
        self._generate_node_ch_reref_types()
        self._generate_node_ch_coords()
        self._generate_node_ch_regions()
        self._generate_node_ch_subregions()
        node_single_hemispheres = self._generate_node_ch_hemispheres()
        self._generate_node_lateralisation(node_single_hemispheres)
        self._generate_node_ch_epoch_orders()

    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results.

        If the types of each channel in a seed/target for a given node are
        identical, this type is given as a string, otherwise the unique types
        are taken and joined into a single string by the " & " characters.
        """
        ch_types = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            types = []
            for single_name in self._comb_names_list[ch_i]:
                types.append(
                    self.signal.data[0].get_channel_types(picks=single_name)[0]
                )
            ch_types[combined_name] = combine_vals_list(unique(types))

        node_ch_types = [[], []]
        groups = ["_seeds_str", "_targets_str"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_types[group_i].append(ch_types[name])
        self.extra_info["node_ch_types"] = node_ch_types

    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity results.

        If the rereferencing types of each channel in a seed/target for a given
        node are identical, this type is given as a string, otherwise the unique
        types are taken and joined into a single string by the " & " characters.
        """
        ch_reref_types = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            reref_types = ordered_list_from_dict(
                list_order=self._comb_names_list[ch_i],
                dict_to_order=self.extra_info["ch_reref_types"],
            )
            unique_types = unique(reref_types)
            ch_reref_types[combined_name] = combine_vals_list(unique_types)

        node_reref_types = [[], []]
        groups = ["_seeds_str", "_targets_str"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_reref_types[group_i].append(ch_reref_types[name])
        self.extra_info["node_ch_reref_types"] = node_reref_types

    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results,
        averaged across for each channel in the seeds and targets."""
        ch_coords = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            ch_coords[combined_name] = np.mean(
                [
                    self.signal.get_coordinates(single_name)[0]
                    for single_name in self._comb_names_list[ch_i]
                ],
                axis=0,
            ).tolist()

        node_ch_coords = [[], []]
        groups = ["_seeds_str", "_targets_str"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_coords[group_i].append(ch_coords[name])
        self.extra_info["node_ch_coords"] = node_ch_coords

    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results.

        If the regions of each channel in a seed/target for a given node are
        identical, this regions is given as a string, otherwise the unique
        regions are taken and joined into a single string by the " & "
        characters.
        """
        ch_regions = {}
        for node_i, combined_name in enumerate(self._comb_names_str):
            regions = ordered_list_from_dict(
                list_order=self._comb_names_list[node_i],
                dict_to_order=self.extra_info["ch_regions"],
            )
            ch_regions[combined_name] = combine_vals_list(unique(regions))

        node_ch_regions = [[], []]
        groups = ["_seeds_str", "_targets_str"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_regions[group_i].append(ch_regions[name])
        self.extra_info["node_ch_regions"] = node_ch_regions

    def _generate_node_ch_subregions(self) -> None:
        """Gets the subregions of channels in the connectivity results.

        If the subregions of each channel in a seed/target for a given node are
        identical, these subregions are given as a string, otherwise the unique
        subregions are taken and joined into a single string by the " & "
        characters.
        """
        ch_subregions = {}
        for node_i, combined_name in enumerate(self._comb_names_str):
            subregions = ordered_list_from_dict(
                list_order=self._comb_names_list[node_i],
                dict_to_order=self.extra_info["ch_subregions"],
            )
            ch_subregions[combined_name] = combine_vals_list(unique(subregions))

        node_ch_subregions = [[], []]
        groups = ["_seeds_str", "_targets_str"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_subregions[group_i].append(ch_subregions[name])
        self.extra_info["node_ch_subregions"] = node_ch_subregions

    def _generate_node_ch_hemispheres(self) -> list[list[bool]]:
        """Gets the hemispheres of channels in the connectivity results.

        If the hemispheres of each channel in a seed/target for a given node are
        identical, this hemispheres is given as a string, otherwise the unique
        hemispheres are taken and joined into a single string by the " & "
        characters.

        RETURNS
        -------
        node_single_hemispheres : list[list[bool]]
        -   list containing two sublists of bools stating whether the channels
            in the seeds/targets of each node were derived from the same
            hemisphere.
        """
        ch_hemispheres = {}
        single_hemispheres = {name: True for name in self._comb_names_str}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            hemispheres = ordered_list_from_dict(
                list_order=self._comb_names_list[ch_i],
                dict_to_order=self.extra_info["ch_hemispheres"],
            )
            unique_types = unique(hemispheres)
            if len(unique_types) > 1:
                single_hemispheres[combined_name] = False
            ch_hemispheres[combined_name] = combine_vals_list(unique_types)

        node_ch_hemispheres = [[], []]
        node_single_hemispheres = [[], []]
        groups = ["_seeds_str", "_targets_str"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_hemispheres[group_i].append(ch_hemispheres[name])
                node_single_hemispheres[group_i].append(
                    single_hemispheres[name]
                )
        self.extra_info["node_ch_hemispheres"] = node_ch_hemispheres

        return node_single_hemispheres

    def _generate_node_lateralisation(
        self, node_single_hemispheres: list[list[bool]]
    ) -> None:
        """Gets the lateralisation of the channels in the connectivity node.

        Can either be "contralateral" if the seed and target are from different
        hemispheres, "ipsilateral" if the seed and target are from the same
        hemisphere, or "ipsilateral & contralateral" if the seed and target are
        from a mix of same and different hemispheres.

        PARAMETERS
        ----------
        node_single_hemispheres : list[list[bool]]
        -   list containing two sublists of bools stating whether the channels
            in the seeds/targets of each node were derived from the same
            hemisphere.
        """
        node_lateralisation = []
        node_ch_hemispheres = self.extra_info["node_ch_hemispheres"]
        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                if (
                    not node_single_hemispheres[0][node_i]
                    or not node_single_hemispheres[1][node_i]
                ):
                    lateralisation = "ipsilateral & contralateral"
                else:
                    lateralisation = "contralateral"
            else:
                lateralisation = "ipsilateral"
            node_lateralisation.append(lateralisation)
        self.extra_info["node_lateralisation"] = node_lateralisation

    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results.

        If either the seed or target has a "shuffled" epoch order, the epoch
        order of the node is "shuffled", otherwise it is "original".
        """
        ch_epoch_orders = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            epoch_orders = ordered_list_from_dict(
                list_order=self._comb_names_list[ch_i],
                dict_to_order=self.extra_info["ch_epoch_orders"],
            )
            ch_epoch_orders[combined_name] = combine_vals_list(
                unique(epoch_orders)
            )

        node_epoch_orders = []
        for seed_name, target_name in zip(self._seeds_str, self._targets_str):
            if (
                ch_epoch_orders[seed_name] == "original"
                and ch_epoch_orders[target_name] == "original"
            ):
                order = "original"
            else:
                order = "shuffled"
            node_epoch_orders.append(order)
        self.extra_info["node_ch_epoch_orders"] = node_epoch_orders

    @abstractmethod
    def save_results(self) -> None:
        """Converts the results and additional information to a dictionary and
        saves them as a file."""

    @abstractmethod
    def results_as_dict(self) -> None:
        """Organises the results and additional information into a
        dictionary."""
