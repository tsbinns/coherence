"""Analyses connectivity results.

METHODS
-------
coherence_analysis
-   Analyses coherence results.
"""


from coh_handle_files import generate_results_fpath, load_file
from coh_postprocess import PostProcess


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

    first_type = True
    for type_i, result_type in enumerate(result_types):
        if type_i > 0:
            first_type = False
        first_result = True
        for result_info in to_analyse.values():
            result_fpath = generate_results_fpath(
                folderpath=results_folderpath,
                subject=result_info["sub"],
                session=result_info["ses"],
                task=result_info["task"],
                acquisition=result_info["acq"],
                run=result_info["run"],
                result_type=result_type,
                filetype=".json",
            )
            result = load_file(fpath=result_fpath)
            if first_type:
                if first_result:
                    results = PostProcess(
                        results=result,
                        extract_from_dicts=extract_from_dicts,
                        identical_entries=identical_entries,
                        discard_entries=discard_entries,
                    )
                    first_result = False
                else:
                    results.append(
                        results=result,
                        extract_from_dicts=extract_from_dicts,
                        identical_entries=identical_entries,
                        discard_entries=discard_entries,
                    )
            else:
                results.merge(
                    results=result,
                    extract_from_dicts=extract_from_dicts,
                    identical_entries=identical_entries,
                    discard_entries=discard_entries,
                    allow_missing=False,
                )
