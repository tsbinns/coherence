"""Analyses connectivity results.

METHODS
-------
coherence_analysis
-   Analyses coherence results.
"""

from coh_postprocess import load_results_of_types


def coherence_analysis(
    results_folderpath: str,
    to_analyse: dict[str],
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

    result_types : list[str]
    -   The types of results to analyse. Includes coherence and the imaginary
        part of coherence by default.
    """

    extract_from_dicts = {
        "metadata": ["sub", "med", "stim", "ses", "task", "run"]
    }
    identical_entries = ["freqs"]
    discard_entries = ["samp_freq", "subject_info", "coherence_dimensions"]

    results = load_results_of_types(
        results_folderpath=results_folderpath,
        to_analyse=to_analyse,
        result_types=result_types,
        extract_from_dicts=extract_from_dicts,
        identical_entries=identical_entries,
        discard_entries=discard_entries,
    )
