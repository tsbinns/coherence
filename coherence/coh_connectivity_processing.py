"""Generates connectivity results from preprocessed data.

METHODS
-------
connectivity_coherence_analysis
-   Takes a coh_signal.Signal object of pre-processed data and analyses the
    coherence (standard or imaginary) between the signals.
"""

from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_connectivity import ConnectivityCoherence
import coh_signal


def coherence_processing(
    signal: coh_signal.Signal,
    folderpath_extras: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    save: bool,
) -> None:
    """
    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_extras : str
    -   The folderpath to the location of the datasets' 'extras', e.g. the
        annotations, processing settings, etc...

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    subject : str
    -   The name of the subject whose data will be analysed.

    session : str
    -   The name of the session for which the data will be analysed.

    task : str
    -   The name of the task for which the data will be analysed.

    acquisition : str
    -   The name of the acquisition mode for which the data will be analysed.

    run : str
    -   The name of the run for which the data will be analysed.

    save : bool
    -   Whether or not to save the results of the analysis.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    analysis_settings_fpath = generate_analysiswise_fpath(
        folderpath_extras + "\\settings", analysis, ".json"
    )
    data_settings_fpath = generate_sessionwise_fpath(
        folderpath_extras,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "settings",
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=analysis_settings_fpath)
    analysis_settings = analysis_settings["connectivity_coherence"]
    data_settings = load_file(fpath=data_settings_fpath)
    data_settings = data_settings["connectivity_coherence"]

    ### Data processing
    ## Coherence analysis
    for method in analysis_settings["methods"]:
        coherence = ConnectivityCoherence(signal)
        coherence.process(
            method=method,
            mode=analysis_settings["mode"],
            seeds=data_settings["seeds"],
            targets=data_settings["targets"],
            fmin=analysis_settings["fmin"],
            fmax=analysis_settings["fmax"],
            fskip=analysis_settings["fskip"],
            faverage=analysis_settings["faverage"],
            tmin=analysis_settings["tmin"],
            tmax=analysis_settings["tmax"],
            mt_bandwidth=analysis_settings["mt_bandwidth"],
            mt_adaptive=analysis_settings["mt_adaptive"],
            mt_low_bias=analysis_settings["mt_low_bias"],
            cwt_freqs=analysis_settings["cwt_freqs"],
            cwt_n_cycles=analysis_settings["cwt_n_cycles"],
            shuffle_group=analysis_settings["shuffle_group"],
            n_shuffles=analysis_settings["n_shuffles"],
            shuffle_rng_seed=analysis_settings["shuffle_rng_seed"],
            average_timepoints=analysis_settings["average_timepoints"],
            block_size=analysis_settings["block_size"],
            n_jobs=analysis_settings["n_jobs"],
        )
        if save:
            coherence_fpath = generate_sessionwise_fpath(
                folderpath_extras,
                dataset,
                subject,
                session,
                task,
                acquisition,
                run,
                f"connectivity-{method}",
                ".json",
            )
            coherence.save_results(coherence_fpath)
