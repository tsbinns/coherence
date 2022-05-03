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
from matplotlib import pyplot as plt
import numpy as np
from coh_exceptions import EntryLengthError
from coh_handle_entries import (
    check_master_entries_in_sublists,
    check_non_repeated_vals_lists,
    check_vals_identical_list,
    combine_col_vals_df,
    get_eligible_idcs_lists,
    get_group_names_idcs,
    reorder_rows_dataframe,
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
            "x_axis_var",
            "y_axis_vars",
            "y_axis_limits_grouping",
            "figure_grouping",
            "subplot_grouping",
            "analysis_entries",
            "identical_entries",
        ]

        present_entries = ["n_from"]
        if self.eligible_values is not None:
            present_entries.extend(list(self.eligible_values.keys()))
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
            if self.var_measure is not None:
                if f"{entry}_{self.var_measure}" in self._results.keys():
                    add_entries.append(f"{entry}_{self.var_measure}")
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
            self._eligible_idcs = np.arange(len(self._results.index)).tolist()

    def _sort_x_axis_limit_idcs(self) -> None:
        """Finds the indices of the x-axis limits"""

        self._x_axis_limit_idcs = {}

        for eligible_idx in self._eligible_idcs:
            x_axis_vals = self._results[self.x_axis_var][eligible_idx]
            if self.x_axis_limits is not None:
                limit_idcs = [
                    x_axis_vals.index(limit) for limit in self.x_axis_limits
                ]
            else:
                limit_idcs = [0, len(x_axis_vals) - 1]
            self._x_axis_limit_idcs[eligible_idx] = limit_idcs

    def _sort_y_axis_limits_grouping(self) -> None:
        """Finds the names and indices of groups that will share y-axis limits.

        RAISES
        ------
        ValueError
        -   Raised if an entry of the grouping factor for the y-axis limits is
            missing from the grouping factors for the figures or subplots.
        """

        self._y_axis_limits_idcs = {}
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
                self._y_axis_limits_idcs = get_group_names_idcs(
                    dataframe=self._results,
                    keys=grouping_entries,
                    eligible_idcs=self._eligible_idcs,
                    replacement_idcs=self._eligible_idcs,
                    special_vals=self._special_values,
                )

        if not self._y_axis_limits_idcs:
            self._y_axis_limits_idcs["ALL"] = deepcopy(self._eligible_idcs)

    def _get_extremes_vars(
        self,
        var_names: list[str],
        extra_vars: Union[list[Union[list[str], None]], None] = None,
        idcs: Union[list[int], None] = None,
        share_across_vars: bool = False,
        min_cap: Union[int, float, None] = None,
        max_cap: Union[int, float, None] = None,
    ) -> dict[list[Union[int, float]]]:
        """Finds the minimum and maximum values within the results for a
        specified set of columns and set of rows.

        PARAMETERS
        ----------
        var_names : list[str]
        -   Names of the column in the results to check.

        extra_vars : list[list[str] | None] | None; default None
        -   List of lists containing the names of the columns in the results
            which should be added to and subtracted from the results,
            respectively, such as standard error or standard deviation measures.
        -   Each list corresponds to the variable in the same position in
            'var_names'.


        idcs : list[int] | None; default None
        -   Indices of the rows in the results to check.
        -   If 'None', all rows are checked.

        share_across_vars : bool; default False
        -   Whether or not to have the minimum and maximum values shared across
            the variables.

        min_cap : int | float | None; default None
        -   Minimum value that can be set.

        max_cap : int | float | None; default None
        -   Maximum value that can be set.

        RETURNS
        -------
        extremes : dict[int | float]
        -   Dictionary where the keys are the variables checked, and the values
            a list with two entries corresponding to the minimum and maximum
            values of the checked results, for the corresponding variable.
        """

        if idcs is None:
            idcs = np.arange(len(self._results.index)).tolist()

        extremes = {}
        for idx, var_name in enumerate(var_names):
            extremes[var_name] = self._get_extremes_var(
                var_name=var_name,
                extra_var=extra_vars[idx],
                idcs=idcs,
                min_cap=min_cap,
                max_cap=max_cap,
            )

        if share_across_vars:
            min_val = min(np.ravel(list(extremes.values())).tolist())
            max_val = max(np.ravel(list(extremes.values())).tolist())
            for var in var_names:
                extremes[var] = [min_val, max_val]

        return extremes

    def _get_extremes_var(
        self,
        var_name: str,
        extra_var: Union[str, None] = None,
        idcs: Union[list[int], None] = None,
        min_cap: Union[int, float, None] = None,
        max_cap: Union[int, float, None] = None,
    ) -> list[Union[int, float]]:
        """Finds the minimum and maximum values within the results for a
        specified column and set of rows.

        PARAMETERS
        ----------
        var_name : str
        -   Name of the column in the results to check.

        extra_var : str | None; default None
        -   Names of the column in the results which should be added to and
            subtracted from the results, respectively, such as standard error
            or standard deviation measures.

        idcs : list[int] | None; default None
        -   Indices of the rows in the results to check.
        -   If 'None', all rows are checked.

        min_cap : int | float | None; default None
        -   Minimum value that can be set.

        max_cap : int | float | None; default None
        -   Maximum value that can be set.

        RETURNS
        -------
        list
        -   List with two entries corresponding to the minimum and maximum
            values of the checked results, respectively.
        """

        if extra_var is None:
            extra_var = []
        if idcs is None:
            idcs = np.arange(len(self._results.index)).tolist()

        min_val = float("inf")
        max_val = float("-inf")

        for row_i in idcs:
            x_lim_idcs = self._x_axis_limit_idcs[row_i]
            main_vals = self._results[var_name][row_i][
                x_lim_idcs[0] : x_lim_idcs[1] + 1
            ]
            minima = [min(main_vals)]
            maxima = [max(main_vals)]
            extra_vals = self._results[extra_var][row_i]
            if extra_vals is not None:
                extra_vals = extra_vals[x_lim_idcs[0] : x_lim_idcs[1] + 1]
                subbed_vals = np.subtract(main_vals, extra_vals).tolist()
                added_vals = np.add(main_vals, extra_vals).tolist()
                minima.append(min(subbed_vals))
                maxima.append(max(added_vals))
            minimum = min(minima)
            maximum = max(maxima)
            if minimum < min_val:
                min_val = minimum
            if maximum > max_val:
                max_val = maximum

        boundary_val = (max_val - min_val) * 0.05
        min_val = min_val - boundary_val
        max_val = max_val + boundary_val

        if min_val < min_cap:
            min_val = min_cap
        if max_val > max_cap:
            max_val = max_cap

        return [min_val, max_val]

    def _sort_y_axis_limits_inputs(self) -> tuple[bool, list[list[str]]]:
        """Sorts inputs for setting the y-axis limits.

        RETURNS
        -------
        share_across_vars : bool
        -   Whether or not the y-axis limits should be shared across all
            variables being plotted on the y-axes.

        extra_vars : list[list[str]]
        -   Additional values to combine with those of the variables being
            plotted on the y-axes (such as standard error or standard deviation
            values) when determining the y-axis limits.
        """

        if self.y_axis_cap_min is None:
            self.y_axis_cap_min = float("-inf")
        if self.y_axis_cap_max is None:
            self.y_axis_cap_max = float("inf")

        share_across_vars = True
        if self.y_axis_limits_grouping is not None:
            if "Y_AXIS_VARS" in self.y_axis_limits_grouping:
                share_across_vars = False

        if self.var_measure is not None:
            extra_vars = [
                f"{var}_{self.var_measure}" for var in self.y_axis_vars
            ]
        else:
            extra_vars = None

        return share_across_vars, extra_vars

    def _sort_y_axis_limits(self) -> None:
        """Checks that the limits for the y-axis variables are in the correct
        format, if provided, or generates the limits if the 'y_axis_limits'
        input is 'None'.

        RAISES
        ------
        KeyError
        -   Raised if the keys of the dictionary in the provided 'y_axis_limits'
            do not match the names of the groups in the
            automatically-generated y-axis limit group indices.
        -   Raised if the keys of the dictionary within each group dictionary do
            not contain the names (and hence limits) for each y-axis variable
            being plotted.
        """

        if self.y_axis_limits is not None:
            all_repeated = check_non_repeated_vals_lists(
                lists=[
                    self.y_axis_limits.keys(),
                    self._y_axis_limits_idcs.keys(),
                ],
                allow_non_repeated=True,
            )
            if not all_repeated:
                raise KeyError(
                    "Error when trying to plot results:\nNames of the groups "
                    "in the specified y-axis limits do not match those "
                    "generated from the results:\n- Provided names: "
                    f"{self.y_axis_limits.keys()}\n- Names should be: "
                    f"{self._y_axis_limits_idcs.keys()}\n"
                )
            for group_name, var_lims in self.y_axis_limits.items():
                for var in self.y_axis_vars:
                    if var not in var_lims.keys():
                        raise KeyError(
                            "Error when trying to plot results:\nMissing "
                            f"limits for the y-axis variable '{var}' in the "
                            f"group '{group_name}'.\n"
                        )
        else:
            self.y_axis_limits = {}
            share_across_vars, extra_vars = self._sort_y_axis_limits_inputs()
            for group_name, idcs in self._y_axis_limits_idcs.items():
                self.y_axis_limits[group_name] = self._get_extremes_vars(
                    var_names=self.y_axis_vars,
                    extra_vars=extra_vars,
                    idcs=idcs,
                    share_across_vars=share_across_vars,
                    min_cap=self.y_axis_cap_min,
                    max_cap=self.y_axis_cap_max,
                )

    def _sort_figure_grouping(self) -> None:
        """Sorts the groups for which indices of rows in the results should be
        plotted on the same set of figures."""

        if self.figure_grouping is not None:
            figure_grouping_entries = [
                entry
                for entry in self.figure_grouping
                if entry != "Y_AXIS_VARS"
            ]
        else:
            figure_grouping_entries = []

        if figure_grouping_entries != []:
            self._plot_grouping = get_group_names_idcs(
                dataframe=self._results,
                keys=figure_grouping_entries,
                eligible_idcs=self._eligible_idcs,
                replacement_idcs=self._eligible_idcs,
                special_vals=self._special_values,
            )
        else:
            self._plot_grouping["ALL"] = self._eligible_idcs

    def _sort_subplot_grouping(self) -> None:
        """Sorts the groups for which indices of rows in the results should be
        plotted on the same subplots on each set of figures."""

        if self.subplot_grouping is not None:
            subplot_grouping_entries = [
                entry
                for entry in self.subplot_grouping
                if entry != "Y_AXIS_VARS"
            ]
        else:
            subplot_grouping_entries = []

        for fig_group, idcs in self._plot_grouping.items():
            if subplot_grouping_entries != []:
                self._plot_grouping[fig_group] = get_group_names_idcs(
                    dataframe=self._results,
                    keys=subplot_grouping_entries,
                    eligible_idcs=self._plot_grouping[fig_group],
                    replacement_idcs=self._plot_grouping[fig_group],
                    special_vals=self._special_values,
                )
            else:
                self._plot_grouping[fig_group] = {"ALL": deepcopy(idcs)}

    def _sort_y_axis_labels(self) -> None:
        """Checks that y-axis labels are in the appropriate format,
        automatically generating them based on the variables being plotted if
        the labels are not specified.

        RAISES
        ------
        EntryLengthError
        -   Raised if the lengths of the y-axis labels does not match the number
            of different variables being plotted on the y-axes. Only raised if
            the y-axis labels are not 'None'.
        -   Raised if multiple variables are being plotted on the same y-axis,
            but more than one y-axis label is given. Only raised if the y-axis
            labels are not 'None'.
        """

        if self.y_axis_labels is None:
            if (
                "Y_AXIS_VARS" in self.figure_grouping
                or "Y_AXIS_VARS" in self.subplot_grouping
            ):
                self.y_axis_labels = self.y_axis_vars
            else:
                self.y_axis_labels = [""]
                self.y_axis_labels.extend(
                    f"{var} / " for var in self.y_axis_vars
                )
                self.y_axis_labels = self.y_axis_labels[:-3]
        else:
            if (
                "Y_AXIS_VARS" in self.figure_grouping
                or "Y_AXIS_VARS" in self.subplot_grouping
            ):
                if len(self.y_axis_labels) != len(self.y_axis_vars):
                    raise EntryLengthError(
                        "Error when trying to plot results:\nThe number of "
                        "different variables being plotted seperately on the "
                        f"y-axes ({len(self.y_axis_vars)}) and the number of "
                        f"y-axis labels ({len(self.y_axis_labels)}) do not "
                        "match."
                    )

            else:
                if len(self.y_axis_labels) != 1:
                    raise EntryLengthError(
                        "Error when tyring to plot results:\nThe different "
                        "variables are being plotted together on the y-axes, "
                        "and so there can only be a single y-axis label, but "
                        f"there are {len(self.y_axis_labels)} labels."
                    )

    def _sort_special_indexing_values(self) -> None:
        """Sorts special values to use when finding rows of results that belong
        to the same group.
        -   If 'average_as_equal' is 'True', converts any string beginning with
            "avg[" (indicating that data has been averaged across multiple
            result types) in the columns into the string "avg_", followed by the
            name of the column, thereby allowing averaged results to be treated
            as belonging to the same type, regardless of the specific types of
            results they were averaged from.
        """

        if self.average_as_equal:
            self._special_values = {"avg[": "avg_"}
        else:
            self._special_values = None

    def _sort_plot_grouping(self) -> None:
        """Sorts the figure and subplot groups, finding the indices of the
        corresponding rows in the results."""

        self._plot_grouping = {}
        self._sort_figure_grouping()
        self._sort_subplot_grouping()

    def _sort_values_order(self) -> None:
        """"""

        if self.order_values is not None:
            self._results = reorder_rows_dataframe(
                dataframe=self._results,
                key=list(self.order_values.keys())[0],
                values_order=list(self.order_values.values())[0],
            )

    def _sort_plot_inputs(self) -> None:
        """Sorts the plotting settings."""

        self._sort_special_indexing_values()
        self._sort_values_order()
        self._discard_entries(entries=self._get_missing_entries())
        self._check_identical_entries()
        self._get_eligible_indices()
        self._sort_x_axis_limit_idcs()
        self._sort_y_axis_limits_grouping()
        self._sort_y_axis_limits()
        self._sort_y_axis_labels()
        self._sort_plot_grouping()

    def _get_figure_title(self, group_name: str) -> str:
        """Generates a title for a figure based on the identical entries in the
        results (if any), and the group of results being plotted on the figure.

        PARAMETERS
        ----------
        group_name : str
        -   Name of the group of results being plotted on the figure.

        RETURNS
        -------
        str
        -   Title of the figure in two lines. Line one: identical entry names
            and values. Line two: group name.
        """

        if self.identical_entries is not None:
            identical_entries_title = combine_col_vals_df(
                dataframe=self._results,
                keys=self.identical_entries,
                idcs=[0],
                special_vals=self._special_values,
            )[0]
        else:
            identical_entries_title = ""

        return f"{identical_entries_title}\n{group_name}"

    def _sort_axes(self, axes: plt.Axes, n_rows: int, n_cols: int) -> plt.Axes:
        """Sorts the pyplot Axes object by adding an additional row and/or
        column if there is only one row and/or column in the object, useful for
        later indexing.

        PARAMETERS
        ----------
        axes : matplotlib pyplot Axes
        -   The axes to sort.

        n_rows : int
        -   Number of rows in the axes.

        n_cols : int
        -   Number of columns in the axes.

        RETURNS
        -------
        axes : matplotlib pyplot Axes
        -   The sorted axes.
        """

        if n_rows == 1 and n_cols == 1:
            axes = np.asarray([[axes]])
        elif n_rows == 1 and n_cols > 1:
            axes = np.vstack((axes, [None, None]))
        elif n_cols == 1 and n_rows > 1:
            axes = np.hstack((axes, [None, None]))

        return axes

    def _get_values_label(
        self, idx: int, var_name: Union[str, None] = None
    ) -> str:
        """Generates a label for a set of values based on the entries of
        different types being plotted on the same subplot.

        PARAMETERS
        ----------
        idx : int
        -   Index of the row of the results for the values being plotted.

        var_name : str | None; default None
        -   Name of the variable being plotted. Optional, as you may not want to
            include the variable name if you are only plotting values from a
            single variable type.

        RETURNS
        -------
        str
        -   Label of the values being plotted.
        """

        label = combine_col_vals_df(
            dataframe=self._results,
            keys=self.analysis_entries,
            idcs=[idx],
            special_vals=self._special_values,
        )[0]

        if var_name is not None:
            label = f"{var_name}: {label}"

        return label

    def _get_vals_to_plot(
        self, y_var: str, idx: int
    ) -> tuple[
        list[Union[int, float]],
        list[Union[int, float]],
        list[list[Union[int, float]]],
    ]:
        """For a given row of results, gets the values to plot on the x- and
        y-axes, plus the boundaries for filling in for the variability measure,
        if applicable.

        PARAMETERS
        ----------
        y_var : str
        -   Variable whose values are being plotted.

        idx : int
        -   Index of the row of the values in the results being plotted.

        RETURNS
        -------
        x_vals : list[int | float]
        -   Values to plot on the x-axis.

        y_vals : list[int | float]
        -   Values to plot on the y-axis.

        var_values : list[list[int | float]]
        -   Y-axis values with the addition of the variability measures. The
            first entry is the values in 'y_vals' minus the corresponding values
            in the variability measure, whilst the second entry is the values in
            'y_vals' plus the corresponding values in the variability measure.
        """

        x_idcs = self._x_axis_limit_idcs[idx]
        x_vals = self._results[self.x_axis_var][idx][x_idcs[0] : x_idcs[1] + 1]
        y_vals = self._results[y_var][idx][x_idcs[0] : x_idcs[1] + 1]

        var_measure = f"{y_var}_{self.var_measure}"
        if (
            self.var_measure is not None
            and self._results[var_measure][idx] is not None
        ):
            var_values = [[], []]
            var_values[0] = np.subtract(
                y_vals,
                self._results[var_measure][idx][x_idcs[0] : x_idcs[1] + 1],
            )
            var_values[1] = np.add(
                y_vals,
                self._results[var_measure][idx][x_idcs[0] : x_idcs[1] + 1],
            )
        else:
            var_values = None

        return x_vals, y_vals, var_values

    def _establish_subplot(
        self,
        subplot: plt.Axes,
        title: str,
        y_axis_vars: list[str],
    ) -> None:
        """Adds a title and axis labels to the subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        title : str
        -   Title of the subplot.

        y_axis_vars : list[str]
        -   Names of the y-axis variables being plotted, used to generate the
            y-axis label.
        """

        subplot.set_title(title)
        subplot.set_xlabel(self.x_axis_label)
        if len(y_axis_vars) == 1:
            y_label = self.y_axis_labels[self.y_axis_vars.index(y_axis_vars[0])]
        else:
            y_label = self.y_axis_labels
        subplot.set_ylabel(y_label)

    def _get_y_axis_limits_group(self, idcs: list[int]) -> str:
        """Finds which y-axis limit group the results of the given indices
        belongs to.

        PARAMETERS
        ----------
        idcs : list[int]
        -   Indices in the results to find their y-axis limit group.

        RETURNS
        -------
        group_name : str
        -   Name of the y-axis limit group.

        RAISES
        ------
        ValueError
        -   Raised if the indices do not belong to any y-axis limit group.
        """

        group_found = False
        for group_name, group_idcs in self._y_axis_limits_idcs.items():
            if set(group_idcs) <= set(self._y_axis_limits_idcs["ALL"]):
                group_found = True
                break

        if not group_found:
            raise ValueError(
                "Error when trying to plot the results:\nThe row indices of "
                f"results {idcs} are not present in any of the y-axis limit "
                "groups."
            )

        return group_name

    def _plot_subplot(
        self,
        subplot: plt.Axes,
        group_name: str,
        group_idcs: list[int],
        y_axis_vars: list[str],
    ) -> None:
        """Plots the results of a specified subplot group belonging to a single
        figure group.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        group_name : str
        -   Name of the subplot group whose results are being plotted.

        group_idcs : list[int]
        -   Indices of the results in the subplot group to plot.

        y_axis_vars : list[str]
        -   Names of the y-axis variables to plot on the same figure group.
        """

        self._establish_subplot(
            subplot=subplot, title=group_name, y_axis_vars=y_axis_vars
        )

        colours = get_plot_colours()
        colour_i = 0
        for idx in group_idcs:
            for y_var in y_axis_vars:
                if len(y_axis_vars) == 1:
                    values_label = self._get_values_label(idx=idx)
                else:
                    values_label = self._get_values_label(
                        idx=idx, var_name=y_var
                    )
                x_vals, y_vals, var_measure_vals = self._get_vals_to_plot(
                    y_var=y_var, idx=idx
                )
                subplot.plot(
                    x_vals,
                    y_vals,
                    color=colours[colour_i],
                    linewidth=2,
                    label=values_label,
                )
                if (
                    self.var_measure is not None
                    and var_measure_vals is not None
                ):
                    subplot.fill_between(
                        x_vals,
                        var_measure_vals[0],
                        var_measure_vals[1],
                        color=colours[colour_i],
                        alpha=0.3,
                    )
                colour_i += 1
        subplot.legend(labelspacing=0)
        y_lim_group = self._get_y_axis_limits_group(idcs=group_idcs)
        subplot.set_ylim(self.y_axis_limits[y_lim_group][y_axis_vars[0]])

    def _establish_figure(
        self, n_rows: int, n_cols: int, title: str
    ) -> tuple[plt.figure, plt.Axes]:
        """Creates a figure with the desired number of subplot rows and columns,
        with a specified title.

        PARAMETERS
        ----------
        n_rows : int
        -   Number of subplot rows in the figure.

        n_cols : int
        -   Number of subplot columns in the figure.

        title : str
        -   Title of the figure.

        RETURNS
        -------
        fig : matplotlib pyplot figure
        -   The created figure.

        axes : matplotlib pyplot Axes
        -   The Axes object of subplots on the figure.
        """

        fig, axes = plt.subplots(n_rows, n_cols)
        plt.tight_layout()
        fig.suptitle(self._get_figure_title(group_name=title))
        axes = self._sort_axes(axes=axes, n_rows=n_rows, n_cols=n_cols)

        return fig, axes

    def _plot_figure(self, group_name: str, y_axis_vars: list[str]) -> None:
        """Plots the results of the subplot groups belonging to a specified
        figure group.

        PARAMETERS
        ----------
        group_name : str
        -   Name of the figure group whose results should be plotted.

        y_axis_vars : list[str]
        -   Names of the y-axis variables to plot on the same figure group.
        """

        if "Y_AXIS_VARS" in self.subplot_grouping:
            subplot_group_names = []
            subplot_group_indices = []
            subplot_group_names.extend(
                [
                    [name] * len(y_axis_vars)
                    for name in self._plot_grouping[group_name].keys()
                ]
            )
            subplot_group_indices.extend(
                [
                    [idcs] * len(y_axis_vars)
                    for idcs in self._plot_grouping[group_name].values()
                ]
            )
        else:
            subplot_group_names = list(self._plot_grouping[group_name].keys())
            subplot_group_indices = list(
                self._plot_grouping[group_name].values()
            )

        n_subplot_groups = len(subplot_group_names)
        n_subplots_per_fig = self.figure_layout[0] * self.figure_layout[1]
        n_figs = int(np.ceil(n_subplot_groups / n_subplots_per_fig))
        n_rows = self.figure_layout[0]
        n_cols = self.figure_layout[1]

        still_to_plot = True
        for fig_i in range(n_figs):
            fig, axes = self._establish_figure(
                n_rows=n_rows, n_cols=n_cols, title=group_name
            )
            subplot_group_i = 0
            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    if still_to_plot:
                        subplot_group_name = subplot_group_names[
                            subplot_group_i
                        ]
                        subplot_group_idcs = subplot_group_indices[
                            subplot_group_i
                        ]
                        self._plot_subplot(
                            subplot=axes[row_i, col_i],
                            group_name=subplot_group_name,
                            group_idcs=subplot_group_idcs,
                            y_axis_vars=y_axis_vars,
                        )
                        subplot_group_i += 1
                        if subplot_group_i == n_subplot_groups:
                            still_to_plot = False
                            extra_subplots = (
                                n_subplots_per_fig * n_figs - n_subplot_groups
                            )
                    else:
                        if extra_subplots > 0:
                            fig.delaxes(axes[row_i, col_i])
            plt.tight_layout()

    def _plot_results(self) -> None:
        """Plots the results of the figure and subplot groups."""

        if self.figure_layout is None:
            self.figure_layout = [1, 1]

        for figure_group in self._plot_grouping.keys():
            if "Y_AXIS_VARS" in self.figure_grouping:
                for y_axis_var in self.y_axis_vars:
                    self._plot_figure(
                        group_name=figure_group,
                        y_axis_vars=[y_axis_var],
                    )
            else:
                self._plot_figure(
                    group_name=figure_group,
                    y_axis_vars=self.y_axis_vars,
                )

    def plot(
        self,
        x_axis_var: str,
        y_axis_vars: list[str],
        x_axis_limits: Union[list[Union[int, float]], None] = None,
        x_axis_label: Union[str, None] = None,
        y_axis_limits: Union[dict[dict[list[Union[int, float]]]], None] = None,
        y_axis_labels: Union[list[str], None] = None,
        y_axis_cap_max: Union[int, float, None] = None,
        y_axis_cap_min: Union[int, float, None] = None,
        var_measure: Union[str, None] = None,
        y_axis_limits_grouping: Union[list[str], None] = None,
        figure_grouping: Union[list[str], None] = None,
        subplot_grouping: Union[list[str], None] = None,
        analysis_entries: Union[list[str], None] = None,
        identical_entries: Union[list[str], None] = None,
        eligible_values: Union[dict[list[str]], None] = None,
        order_values: Union[dict[str], None] = None,
        average_as_equal: bool = True,
        figure_layout: Union[list[int], None] = None,
    ) -> None:
        """Plots the results as line graphs."""

        self.x_axis_var = x_axis_var
        self.y_axis_vars = y_axis_vars
        self.x_axis_limits = x_axis_limits
        self.x_axis_label = x_axis_label
        self.y_axis_limits = y_axis_limits
        self.y_axis_labels = y_axis_labels
        self.y_axis_cap_max = y_axis_cap_max
        self.y_axis_cap_min = y_axis_cap_min
        self.var_measure = var_measure
        self.y_axis_limits_grouping = y_axis_limits_grouping
        self.figure_grouping = figure_grouping
        self.subplot_grouping = subplot_grouping
        self.analysis_entries = analysis_entries
        self.identical_entries = identical_entries
        self.eligible_values = eligible_values
        self.order_values = order_values
        self.figure_layout = figure_layout
        self.average_as_equal = average_as_equal

        self._sort_plot_inputs()

        self._plot_results()

        print("jeff")


def get_plot_colours() -> list[str]:
    """Returns the matplotlib default set of ten colours.

    RETURNS
    -------
    colours : list[str]
    -   The ten default matplotlib colours.
    """

    colours = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    return colours
