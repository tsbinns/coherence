"""Analyses connectivity results.

METHODS
-------
coherence_analysis
-   Analyses coherence results.
"""

from coh_handle_files import generate_fpath_from_analysed
from coh_process_results import load_results_of_types
from coh_settings import get_analysis_settings


def connectivity_analysis(
    folderpath_processing: str,
    folderpath_analysis: str,
    analysis: str,
    save: bool,
) -> None:
    """Analyses coherence results.

    PARAMETERS
    ----------
    folderpath_processing : str
    -   Folderpath to the location where the results of the data processing are
        located.

    folderpath_analysis : str
    -   Folderpath to the location where the analysis settings are located and
        where the output of the analysis should be saved, if applicable.

    analysis : str
    -   Name of the analysis being performed, used for loading the analysis
        settings and, optionally, saving the output of the analysis.

    save : bool
    -   Whether or not to save the output of the analysis.
    """

    ## Loads the analysis settings
    (
        analysis_settings,
        extract_from_dicts,
        identical_entries,
        discard_entries,
    ) = get_analysis_settings(
        settings_fpath=(f"{folderpath_analysis}\\Settings\\{analysis}.json")
    )
    to_analyse = analysis_settings["to_analyse"]
    result_types = analysis_settings["result_types"]
    steps = analysis_settings["steps"]
    band_measures = analysis_settings["freq_band_measures"]
    var_measures = analysis_settings["var_measures"]
    freq_bands = analysis_settings["freq_bands"]

    results = load_results_of_types(
        folderpath_processing=f"{folderpath_processing}\\Data",
        to_analyse=to_analyse,
        result_types=result_types,
        extract_from_dicts=extract_from_dicts,
        identical_entries=identical_entries,
        discard_entries=discard_entries,
    )

    if freq_bands is not None:
        results.freq_band_results(
            bands=freq_bands, attributes=result_types, measures=band_measures
        )

    for step in steps:
        if freq_bands is not None:
            step["identical_keys"].extend(["fband_labels", "fband_freqs"])
        if step["method"] == "average":
            results.average(
                over_key=step["over_key"],
                data_keys=step["data_keys"],
                group_keys=step["group_keys"],
                over_entries=step["over_entries"],
                identical_keys=step["identical_keys"],
                var_measures=var_measures,
            )

    if save:
        results_fpath = generate_fpath_from_analysed(
            analysed=to_analyse,
            parent_folderpath=f"{folderpath_analysis}\\Results",
            analysis=analysis,
            ftype="json",
        )
        results.save_results(fpath=results_fpath)
