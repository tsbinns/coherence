"""Plots connectivity results.

METHODS
-------
connectivity_plotting
-   Plots connectivity results, including coherence, multivariate connectivity,
    and time-reversed Granger causality.
"""

from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_plotting import LinePlot


def connectivity_plotting(
    folderpath_analysis: str,
    folderpath_plotting: str,
    plotting: str,
    save: bool,
) -> None:
    """Plots connectivity results.

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

    save : bool
    -   Whether or not to save the plots.
    """

    ### Plotting setup
    ## Gets the relevant filepaths and loads the plotting settings
    plotting_settings_fpath = generate_analysiswise_fpath(
        folderpath_plotting + "\\Settings", plotting, ".json"
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
            plot.plot()
        else:
            raise ValueError(
                "Error when trying to plot the analysis results:\nThe plot "
                f"type '{plot['plot_type']}' is not supported. Only plots of "
                f"type(s) {supported_plot_types} are supported.\n"
            )
        if save:
            figures_fpath = generate_sessionwise_fpath(
                f"{folderpath_plotting}\\Figures",
                analysis_settings["cohort"],
                analysis_settings["sub"],
                analysis_settings["ses"],
                analysis_settings["task"],
                analysis_settings["acq"],
                analysis_settings["run"],
                plotting,
                ".svg",
            )
            plot.save(fpath=figures_fpath)