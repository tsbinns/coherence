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
from typing import Union
from coh_handle_entries import sort_inputs_results, dict_to_df


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
            if isinstance(entry, list):
                present_entries.append([*entry])
            else:
                present_entries.append(entry)

        for entry in present_entries:
            if f"{entry}_dimensions" in self._results.keys():
                present_entries.append(entry)
            for measure in self.var_measures:
                if f"{entry}_{measure}" in self._results.keys():
                    present_entries.append(entry)

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
        missing_entries = []

        return [
            missing_entries.append(entry)
            for entry in self._results.keys()
            if entry not in present_entries
        ]

    def _discard_entries(self, entries: list[str]) -> None:
        """Drops entries from the results DataFrame and resets the DataFrame
        index."""

        self._results = self._results.drop(columns=entries)
        self._results = self._results.reset_index()

    def plot(
        self,
        x_axis: str,
        y_axis: list[str],
        x_axis_limits: Union[list[Union[int, float]], None] = None,
        x_axis_label: Union[str, None] = None,
        y_axis_limits: Union[list[list[Union[int, float]]], None] = None,
        y_axis_label: Union[list[str], None] = None,
        var_measures: Union[list[str], None] = None,
        y_axis_grouping: Union[list[str], None] = None,
        figure_grouping: Union[list[str], None] = None,
        subplot_grouping: Union[list[str], None] = None,
        analysis_entries: Union[list[str], None] = None,
        identical_entries: Union[list[str], None] = None,
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
        self.y_axis_grouping = y_axis_grouping
        self.figure_grouping = figure_grouping
        self.subplot_grouping = subplot_grouping
        self.analysis_entries = analysis_entries
        self.identical_entries = identical_entries
        self.average_as_equal = average_as_equal
        self.figure_layout = figure_layout

        missing_entries = self._get_missing_entries()
        self._discard_entries(entries=missing_entries)

        ### CHECK IDENTICAL ENTRIES ARE IDENTICAL

        ### FIND Y-AXIS LIMITS

        ### FIND GROUPINGS
        ### -   FIGURE GROUPINGS
        ### -       SUBPLOT GROUPINGS

        ### PLOT RESULTS
