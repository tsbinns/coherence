"""Plots power results."""

from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_plotting import LinePlot


def power_standard_plotting(
    folderpath_analysis: str, folderpath_plotting: str, plotting: str
) -> None:
    """Plots standard power results.

    PARAMETERS
    ----------
    folderpath_analysis : str
    -   Folderpath to the location where the analysed results are located.

    folderpath_plotting : str
    -   Folderpath to the location where the plotting settings are located and
        where the plots should be saved, if applicable.

    plotting : str
    -   Name of the plotting being performed, used for loading the plotting
        settings and, optionally, saving the plots.
    """
    ### Plotting setup
    ## Gets the relevant filepaths and loads the plotting settings
    plotting_settings_fpath = generate_analysiswise_fpath(
        f"{folderpath_plotting}\\Settings", plotting, ".json"
    )
    plotting_settings = load_file(fpath=plotting_settings_fpath)
    analysis_settings = plotting_settings["analysis"]
    results_fpath = generate_sessionwise_fpath(
        f"{folderpath_analysis}\\Results",
        analysis_settings["cohort"],
        analysis_settings["sub"],
        analysis_settings["ses"],
        analysis_settings["task"],
        analysis_settings["acq"],
        analysis_settings["run"],
        plotting,
        ".json",
    )

    ## Loads the results to plot
    results = load_file(fpath=results_fpath)

    ### Plotting
    supported_plot_types = ["line_plot"]
    for plot_settings in plotting_settings["plotting"]:
        if plot_settings["type"] == "line_plot":
            plot = LinePlot(results=results)
            plot.plot(
                x_axis_var=plot_settings["x_axis_var"],
                y_axis_vars=plot_settings["y_axis_vars"],
                x_axis_limits=plot_settings["x_axis_limits"],
                x_axis_label=plot_settings["x_axis_label"],
                y_axis_limits=plot_settings["y_axis_limits"],
                y_axis_labels=plot_settings["y_axis_labels"],
                y_axis_cap_max=plot_settings["y_axis_cap_max"],
                y_axis_cap_min=plot_settings["y_axis_cap_min"],
                var_measure=plot_settings["var_measure"],
                y_axis_limits_grouping=plot_settings["y_axis_limits_grouping"],
                figure_grouping=plot_settings["figure_grouping"],
                subplot_grouping=plot_settings["subplot_grouping"],
                analysis_keys=plot_settings["analysis_keys"],
                identical_keys=plot_settings["identical_keys"],
                eligible_values=plot_settings["eligible_values"],
                order_values=plot_settings["order_values"],
                figure_layout=plot_settings["figure_layout"],
                average_as_equal=plot_settings["average_as_equal"],
                save=plot_settings["save"],
                save_folderpath=f"{folderpath_plotting}\\Figures\\{plotting}",
                save_ftype=plot_settings["save_ftype"],
            )
        else:
            raise ValueError(
                "Error when trying to plot the analysis results:\nThe plot "
                f"type '{plot['plot_type']}' is not supported. Only plots of "
                f"type(s) {supported_plot_types} are supported.\n"
            )
