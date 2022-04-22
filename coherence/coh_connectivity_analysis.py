"""Analyses connectivity results.

METHODS
-------
coherence_analysis
-   Analyses coherence results.
"""

from coh_handle_files import generate_fpath_from_analysed, load_file
from coh_process_results import load_results_of_types


def connectivity_analysis(
    folderpath_processing: str,
    folderpath_analysis: str,
    analysis: str,
    save: bool = True,
) -> None:
    """Analyses coherence results.

    PARAMETERS
    ----------
    results_folderpath : str
    -   Folderpath to where the results are located.

    to_analyse : dict[str]
    -   Dictionary in which each entry represents a different piece of results.
    -   Contains the keys: 'sub' (subject ID); 'ses' (session name); 'task'
        (task name); 'acq' (acquisition type); and 'run' (run number).

    analysis_steps : list[list]
    -   Instructions for how to analyse the results.
    -   Each entry should be a list with three entries, where: the first is the
        type of processing to apply; the second is the attribute of the results
        to apply this to; and the third specifies which values selected in the
        second entry should be processed.
    -   E.g. ["average", "runs", "ALL"] would average the results across all
        runs. ["average", "ch_types", ["dbs"]] would average the results across
        all channels of type 'dbs'.
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


def get_analysis_settings(
    settings_fpath: str,
) -> tuple[dict, dict, list[str], list[str]]:
    """Gets the default settings for results analysis, as well as those specific
    for the requested analysis.

    PARAMETERS
    ----------
    settings_fpath : str
    -   Filepath to the analysis-specific settings.

    RETURNS
    -------
    analysis_settings : dict
    -   The analysis-specific settings.

    extract_from_dicts : dict[list[str]]
    -   The entries of dictionaries within the results to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the results dictionary.

    identical_entries : list[str]
    -   The entries in the results which are identical across channels and for
        which only one copy is present.

    discard_entries : list[str]
    -   The entries in the results which should be discarded immediately without
        processing.
    """

    analysis_settings = load_file(settings_fpath)

    extract_from_dicts = {
        "metadata": ["sub", "med", "stim", "ses", "task", "run"]
    }

    result_types_dims = [
        f"{results_type}_dimensions"
        for results_type in analysis_settings["result_types"]
    ]
    """
    identical_entries = ["freqs", *result_types_dims]
    discard_entries = [
        "samp_freq",
        "subject_info",
    ]
    """
    identical_entries = ["freqs"]
    discard_entries = ["samp_freq", "subject_info", *result_types_dims]

    return (
        analysis_settings,
        extract_from_dicts,
        identical_entries,
        discard_entries,
    )
