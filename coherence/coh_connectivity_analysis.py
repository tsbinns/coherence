"""Analyses connectivity results.

METHODS
-------
coherence_analysis
-   Analyses coherence results.
"""

from coh_handle_entries import drop_from_list
from coh_handle_files import load_file
from coh_process_results import load_results_of_types


def coherence_analysis(
    results_folderpath: str,
    settings_folderpath: str,
    analysis: str,
    result_types: list[str] = ["connectivity-coh", "connectivity-imcoh"],
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

    result_types : list[str]
    -   The types of results to analyse. Includes coherence and the imaginary
        part of coherence by default.
    """

    ## Loads the analysis settings
    (
        analysis_settings,
        extract_from_dicts,
        identical_entries,
        discard_entries,
    ) = get_analysis_settings(
        settings_fpath=(settings_folderpath + "\\" + analysis + ".json"),
        result_types=result_types,
    )
    to_analyse = analysis_settings["to_analyse"]
    steps = analysis_settings["steps"]
    band_measures = analysis_settings["freq_band_measures"]
    var_measures = analysis_settings["var_measures"]
    freq_bands = analysis_settings["freq_bands"]

    results = load_results_of_types(
        results_folderpath=results_folderpath,
        to_analyse=to_analyse,
        result_types=result_types,
        extract_from_dicts=extract_from_dicts,
        identical_entries=identical_entries,
        discard_entries=discard_entries,
    )

    identical_keys = ["freqs", "seed_coords", "target_coords"]

    if freq_bands is not None:
        results.freq_band_results(
            bands=freq_bands, attributes=result_types, measures=band_measures
        )
        identical_keys.extend(["fband_labels", "fband_freqs"])

    for step in steps:
        method = step[0]
        attribute = step[1]
        entries = step[2]

        group_keys = drop_from_list(
            obj=results.attributes,
            drop=[attribute, *result_types, *identical_keys],
        )

        if method == "average":
            results.average(
                over_key=attribute,
                data_keys=result_types,
                group_keys=group_keys,
                over_entries=entries,
                identical_keys=identical_keys,
                var_measures=var_measures,
            )


def get_analysis_settings(
    settings_fpath: str, result_types: list[str]
) -> tuple[dict, dict, list[str], list[str]]:
    """Gets the default settings for results analysis, as well as those specific
    for the requested analysis.

    PARAMETERS
    ----------
    settings_fpath : str
    -   Filepath to the analysis-specific processing settings.

    result_types : list[str]
    -   Names of the results to analyse.

    RETURNS
    -------
    analysis_settings : dict
    -   The analysis-specific processing settings.

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
    identical_entries = ["freqs"]
    discard_entries = [
        "samp_freq",
        "subject_info",
        *[f"{results_type}_dimensions" for results_type in result_types],
    ]

    return (
        analysis_settings,
        extract_from_dicts,
        identical_entries,
        discard_entries,
    )
