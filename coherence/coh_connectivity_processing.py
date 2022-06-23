"""Generates connectivity results from preprocessed data."""

import os
import numpy as np
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_connectivity import (
    ConnectivityCoherence,
    ConnectivityGranger,
    ConnectivityMultivariate,
)
import coh_signal


def coherence_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    save: bool,
) -> None:
    """Peforms processing to generate, and optionally save, coherence results.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
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
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )
    specific_settings_fpath = generate_sessionwise_fpath(
        os.path.join(folderpath_processing, "Settings", "Specific"),
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        analysis,
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)
    data_settings = load_file(fpath=specific_settings_fpath)

    ### Data processing
    ## Coherence analysis
    if analysis_settings["cwt_freqs"] is not None:
        cwt_freqs = np.arange(
            analysis_settings["cwt_freqs"][0],
            analysis_settings["cwt_freqs"][1] + 1,
        )
    for method in analysis_settings["con_methods"]:
        coherence = ConnectivityCoherence(signal)
        coherence.process(
            con_method=method,
            pow_method=analysis_settings["pow_method"],
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
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=analysis_settings["cwt_n_cycles"],
            average_windows=analysis_settings["average_windows"],
            average_timepoints=analysis_settings["average_timepoints"],
            block_size=analysis_settings["block_size"],
            n_jobs=analysis_settings["n_jobs"],
        )
        if save:
            coherence_fpath = generate_sessionwise_fpath(
                os.path.join(folderpath_processing, "Data"),
                dataset,
                subject,
                session,
                task,
                acquisition,
                run,
                f"connectivity-{method}_{analysis}",
                ".json",
            )
            coherence.save_results(coherence_fpath)


def multivariate_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    save: bool,
) -> None:
    """Peforms processing to generate, and optionally save, multivariate
    conectivity results.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
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
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )
    specific_settings_fpath = generate_sessionwise_fpath(
        os.path.join(folderpath_processing, "Settings", "Specific"),
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        analysis,
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)
    data_settings = load_file(fpath=specific_settings_fpath)

    ### Data processing
    ## Multivariate connectivity analysis
    if analysis_settings["cwt_freqs"] is not None:
        cwt_freqs = np.arange(
            analysis_settings["cwt_freqs"][0],
            analysis_settings["cwt_freqs"][1] + 1,
        )
    for con_method in analysis_settings["con_methods"]:
        multivariate = ConnectivityMultivariate(signal)
        multivariate.process(
            con_method=con_method,
            cohy_method=analysis_settings["cohy_method"],
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
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=analysis_settings["cwt_n_cycles"],
            average_windows=analysis_settings["average_windows"],
            return_topographies=analysis_settings["return_topographies"],
            block_size=analysis_settings["block_size"],
            n_jobs=analysis_settings["n_jobs"],
        )
        if save:
            multivariate_fpath = generate_sessionwise_fpath(
                os.path.join(folderpath_processing, "Data"),
                dataset,
                subject,
                session,
                task,
                acquisition,
                run,
                f"connectivity-{con_method}_{analysis}",
                "",
            )
            multivariate.save_results(multivariate_fpath, "json")


def granger_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    save: bool,
) -> None:
    """Peforms processing to generate, and optionally save, Granger causality
    results.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
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
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )
    specific_settings_fpath = generate_sessionwise_fpath(
        os.path.join(folderpath_processing, "Settings", "Specific"),
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        analysis,
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)
    data_settings = load_file(fpath=specific_settings_fpath)

    ### Data processing
    ## Coherence analysis
    if analysis_settings["cwt_freqs"] is not None:
        cwt_freqs = np.arange(
            analysis_settings["cwt_freqs"][0],
            analysis_settings["cwt_freqs"][1] + 1,
        )
    for gc_method in analysis_settings["gc_methods"]:
        coherence = ConnectivityGranger(signal)
        coherence.process(
            gc_method=gc_method,
            cs_method=analysis_settings["cs_method"],
            seeds=data_settings["seeds"],
            targets=data_settings["targets"],
            n_lags=analysis_settings["n_lags"],
            tmin=analysis_settings["tmin"],
            tmax=analysis_settings["tmax"],
            average_windows=analysis_settings["average_windows"],
            n_jobs=analysis_settings["n_jobs"],
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=analysis_settings["cwt_n_cycles"],
            cwt_use_fft=analysis_settings["cwt_use_fft"],
            cwt_decim=analysis_settings["cwt_decim"],
            mt_bandwidth=analysis_settings["mt_bandwidth"],
            mt_adaptive=analysis_settings["mt_adaptive"],
            mt_low_bias=analysis_settings["mt_low_bias"],
            fmt_fmin=analysis_settings["fmt_fmin"],
            fmt_fmax=analysis_settings["fmt_fmax"],
            fmt_n_fft=analysis_settings["fmt_n_fft"],
        )
        if save:
            granger_fpath = generate_sessionwise_fpath(
                os.path.join(folderpath_processing, "Data"),
                dataset,
                subject,
                session,
                task,
                acquisition,
                run,
                f"connectivity-{gc_method}_{analysis}",
                ".json",
            )
            coherence.save_results(granger_fpath)
