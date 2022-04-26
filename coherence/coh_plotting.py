"""Classes for plotting results.

CLASSES
-------
Plotting : Abstract Base Class
-   Abstract class for plotting results.

LinePlot : subclass of Plotting
-   Class for plotting results on line plots.

BoxPlot : subclass of Plotting
-   Class for plotting results on box plots.

SurfacePlot : subclass of Plotting
-   Class for plotting results on cortical surfaces.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union
import numpy as np
from coh_exceptions import EntryLengthError
from coh_handle_entries import (
    check_master_entries_in_sublists,
    check_vals_identical_list,
    combine_col_vals_df,
    get_eligible_idcs_lists,
    get_group_idcs,
    sort_inputs_results,
    dict_to_df,
)


class Plotting(ABC):
    """Abstract class for plotting results.

    PARAMETERS
    ----------
    results : dict
    -   A dictionary containing results to process.
    -   The entries in the dictionary should be either lists, numpy arrays, or
        dictionaries.
    -   Entries which are dictionaries will have their values treated as being
        identical for all values in the 'results' dictionary, given they are
        extracted from these dictionaries into the results.
    -   Keys ending with "_dimensions" are treated as containing information
        about the dimensions of other attributes in the results, e.g.
        'X_dimensions' would specify the dimensions for attribute 'X'. The
        dimensions should be a list of strings containing the values "channels"
        and "frequencies" in the positions corresponding to the axis of these
        dimensions in 'X'. A single list should be given, i.e. 'X_dimensions'
        should hold for all entries of 'X'.If no dimensions, are given, the 0th
        axis is assumed to correspond to channels and the 1st axis to
        frequencies.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        ['epochs', 'channels', 'frequencies', 'timepoints'], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions ["channels", "frequencies", "epochs", "timepoints"].

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_entries : list[str] | None; default None
    -   The entries in 'results' which are identical across channels and for
        which only one copy is present.
    -   If any dimension attributes are present, these should be included as an
        identical entry, as they will be added automatically.

    discard_entries : list[str] | None; default None
    -   The entries which should be discarded immediately without processing.

    verbose : bool; default True
    -   Whether or not to print updates about the plotting process.

    METHODS
    -------
    """

    def __init__(
        self,
        results: dict,
        extract_from_dicts: Union[dict[list[str]], None] = None,
        identical_entries: Union[list[str], None] = None,
        discard_entries: Union[list[str], None] = None,
        verbose: bool = True,
    ) -> None:

        # Initialises inputs of the object.
        results = sort_inputs_results(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_entries=identical_entries,
            discard_entries=discard_entries,
            verbose=verbose,
        )
        self._results = dict_to_df(obj=results)
        self._verbose = verbose

    @abstractmethod
    def plot(self) -> None:
        """Abstract method for plotting the results."""


class LinePlot(Plotting):
    """Class for plotting results on line plots."""

    def _check_lengths_plot_inputs(self) -> None:
        """Checks that plotting settings are in the appropriate format.

        RAISES
        ------
        EntryLengthError
        -   Raised if the lengths of the y-axis labels does not match the number
            of different variables being plotted on the y-axes. Only raised if
            the y-axis labels is also not 'None'.
        """

        if len(self.y_axis) != len(self.y_axis_label):
            if self.y_axis_limits is not None:
                raise EntryLengthError(
                    "Error when trying to plot the results:\nThe number of "
                    "different variables being plotted on the y-axes "
                    f"({len(self.y_axis)}) and the number of y-axis labels "
                    f"({len(self.y_axis_label)}) do not match."
                )

    def _get_present_entries(self) -> list[str]:
        """Finds which entries in the results have been accounted for in the
        plotting settings.

        RETURNS
        -------
        present_entries : list[str]
        -   Names of the entries in the results accounted for by the plotting
            settings.
        """

        entry_inputs = [
            "x_axis",
            "y_axis",
            "y_axis_grouping",
            "figure_grouping",
            "subplot_grouping",
            "analysis_entries",
            "identical_entries",
        ]

        present_entries = ["n_from"]
        for entry_name in entry_inputs:
            entry = getattr(self, entry_name)
            if entry is not None:
                if isinstance(entry, list):
                    present_entries.extend(entry)
                else:
                    present_entries.append(entry)

        add_entries = []
        for entry in present_entries:
            if f"{entry}_dimensions" in self._results.keys():
                add_entries.append(f"{entry}_dimensions")
            for measure in self.var_measures:
                if f"{entry}_{measure}" in self._results.keys():
                    add_entries.append(f"{entry}_{measure}")
        present_entries.extend(add_entries)

        return present_entries

    def _get_missing_entries(self) -> list[str]:
        """Finds which entries in the results are not accounted for in the
        plotting settings.

        RETURNS
        -------
        list[str]
        -   Names of entries in the reuslts not accounted for by the plotting
            settings.
        """

        present_entries = self._get_present_entries()

        return [
            entry
            for entry in self._results.keys()
            if entry not in present_entries
        ]

    def _discard_entries(self, entries: list[str]) -> None:
        """Drops entries from the results DataFrame and resets the DataFrame
        index."""

        self._results = self._results.drop(columns=entries)
        self._results = self._results.reset_index()

    def _check_identical_entries(self) -> None:
        """Checks that entries in the results marked as identical are
        identical."""

        for entry in self.identical_entries:
            values = deepcopy(self._results[entry])
            if self.average_as_equal:
                for i, val in enumerate(values):
                    if isinstance(val, str):
                        if val[:4] == "avg[":
                            values[i] = f"avg_{entry}"
            is_identical, vals = check_vals_identical_list(to_check=values)
            if not is_identical:
                raise ValueError(
                    "Error when trying to plot the results:\nThe results entry "
                    f"'{entry}' is marked as an identical entry, however its "
                    "values are not identical for all results:\n- Unique "
                    f"values: {vals}\n"
                )

    def _get_eligible_indices(self) -> None:
        """Finds which indices in the results contain values designated as
        eligible for plotting."""

        if self.eligible_values is not None:
            to_check = {
                key: self._results[key] for key in self.eligible_values.keys()
            }
            self._eligible_idcs = get_eligible_idcs_lists(
                to_check=to_check, eligible_vals=self.eligible_values
            )
        else:
            self._eligible_idcs = range(len(self._results.index))

    def _sort_y_axis_limits_grouping(self) -> None:
        """Finds the names and indices of groups that will share y-axis limits.

        RAISES
        ------
        ValueError
        -   Raised if an entry of the grouping factor for the y-axis limits is
            missing from the grouping factors for the figures or subplots.
        """

        self._y_axis_limits_grouping = {}
        if self.y_axis_limits_grouping is not None:
            all_present, absent_entries = check_master_entries_in_sublists(
                master_list=self.y_axis_limits_grouping,
                sublists=[self.figure_grouping, self.subplot_grouping],
                allow_duplicates=False,
            )
            if not all_present:
                raise ValueError(
                    "Error when tyring to plot results:\nThe entry(ies) in the "
                    f"results {self.y_axis_limits_grouping} for creating "
                    "groups that will share the y-axis limits must also be "
                    "accounted for in the entry(ies) for creating groups that "
                    "will be plotted on the same figures/subplots, as plotting "
                    "results with multiple y-axes on the same plot is not yet "
                    "supported.\nThe following entries are unaccounted for: "
                    f"{absent_entries}\n"
                )
            grouping_entries = [
                entry
                for entry in self.y_axis_limits_grouping
                if entry != "Y_AXIS_VARS"
            ]
            if grouping_entries != []:
                combined_vals = combine_col_vals_df(
                    dataframe=self._results,
                    keys=grouping_entries,
                    idcs=self._eligible_idcs,
                    special_vals={"avg[": "avg_"},
                )
                group_idcs, group_names = get_group_idcs(
                    vals=combined_vals, replacement_idcs=self._eligible_idcs
                )
                for idx, name in enumerate(group_names):
                    self._y_axis_limits_grouping[name] = group_idcs[idx]

        if not self._y_axis_limits_grouping:
            self._y_axis_limits_grouping["ALL"] = self._eligible_idcs

    def _get_limits_of_entry(
        self, entry: str
    ) -> tuple[Union[int, float], Union[int, float]]:
        """"""

        max_val = float("-inf")
        min_val = float("inf")
        for idx, values in enumerate(self._results[entry]):
            max_values = deepcopy(values)
            min_values = deepcopy(values)
            if self.var_measures is not None:
                for measure in self.var_measures:
                    if f"{entry}_{measure}" in self._results.keys():
                        max_values += self._results[f"{entry}_{measure}"][idx]
                        max_values -= self._results[f"{entry}_{measure}"][idx]
                        if max(max_values) > max_val:
                            max_val = max(max_values)
                        if min(min_values) < min_val:
                            min_val = min(min_values)
            else:
                if max(max_values) > max_val:
                    max_val = max(max_values)
                if min(min_values) < min_val:
                    min_val = min(min_values)

        return max_val, min_val

    def _sort_y_axis_limits(self) -> None:
        """Finds the y axis limits for the plots"""

        if self.y_axis_limits_grouping is None:
            maximum = float("-inf")
            minimum = float("inf")
            for entry in self.y_axis:
                max_val, min_val = self._get_limits_of_entry(entry=entry)
                if max_val > maximum:
                    maximum = max_val
                if min_val < minimum:
                    minimum = min_val
            self.y_axis_limits = [[minimum, maximum]] * len(self.y_axis)
        elif self.y_axis_limits_grouping == "Y_AXIS":
            self.y_axis_limits = []
            for entry in self.y_axis:
                max_val, min_val = self._get_limits_of_entry(entry=entry)
                self.y_axis_limits.append([min_val, max_val])

    def _sort_plot_inputs(self) -> None:
        """Sorts the plotting settings."""

        self._check_lengths_plot_inputs()
        self._discard_entries(entries=self._get_missing_entries())
        self._check_identical_entries()
        self._get_eligible_indices()
        self._sort_y_axis_limits_grouping()
        self._sort_y_axis_limits()

        ### FIND GROUPINGS
        ### -   FIGURE GROUPINGS
        ### -       SUBPLOT GROUPINGS

    def plot(
        self,
        x_axis: str,
        y_axis: list[str],
        x_axis_limits: Union[list[Union[int, float]], None] = None,
        x_axis_label: Union[str, None] = None,
        y_axis_limits: Union[list[list[Union[int, float]]], None] = None,
        y_axis_label: Union[list[str], None] = None,
        var_measures: Union[list[str], None] = None,
        y_axis_limits_grouping: Union[list[str], None] = None,
        figure_grouping: Union[list[str], None] = None,
        subplot_grouping: Union[list[str], None] = None,
        analysis_entries: Union[list[str], None] = None,
        identical_entries: Union[list[str], None] = None,
        eligible_values: Union[dict[list[str]], None] = None,
        average_as_equal: bool = True,
        figure_layout: Union[list[int], None] = None,
    ) -> None:
        """Plots the results as line graphs."""

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_axis_limits = x_axis_limits
        self.x_axis_label = x_axis_label
        self.y_axis_limits = y_axis_limits
        self.y_axis_label = y_axis_label
        self.var_measures = var_measures
        self.y_axis_limits_grouping = y_axis_limits_grouping
        self.figure_grouping = figure_grouping
        self.subplot_grouping = subplot_grouping
        self.analysis_entries = analysis_entries
        self.identical_entries = identical_entries
        self.eligible_values = eligible_values
        self.average_as_equal = average_as_equal
        self.figure_layout = figure_layout

        self._sort_plot_inputs()

        ### PLOT RESULTS

        print("jeff")
