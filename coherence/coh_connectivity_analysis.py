"""Analyses connectivity results.

METHODS
-------
coherence_analysis
-   Analyses coherence results.
"""

from coh_handle_entries import drop_from_list
from coh_process_results import load_results_of_types


def coherence_analysis(
    results_folderpath: str,
    to_analyse: dict[str],
    analysis_steps: list[list],
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

    extract_from_dicts = {
        "metadata": ["sub", "med", "stim", "ses", "task", "run"]
    }
    identical_entries = ["freqs"]
    discard_entries = [
        "samp_freq",
        "subject_info",
        *[f"{results_type}_dimensions" for results_type in result_types],
    ]

    results = load_results_of_types(
        results_folderpath=results_folderpath,
        to_analyse=to_analyse,
        result_types=result_types,
        extract_from_dicts=extract_from_dicts,
        identical_entries=identical_entries,
        discard_entries=discard_entries,
    )

    identical_keys = ["freqs", "seed_coords", "target_coords"]
    var_measures = ["sem"]
    for analysis in analysis_steps:
        method = analysis[0]
        attribute = analysis[1]
        entries = analysis[2]

        group_keys = drop_from_list(
            obj=results.as_df().keys(),
            drop=[attribute, *result_types, *identical_keys],
        )

        if method == "average":
            results.average(
                over_key=attribute,
                group_keys=group_keys,
                over_entries=entries,
                identical_keys=identical_keys,
                var_measures=var_measures,
            )
