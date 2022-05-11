"""Performs power analysis on pre-processed data.

METHODS
-------
morlet_analysis
-   Takes a coh_signal.Signal object of pre-processed data and performs Morlet
    wavelet power analysis.

fooof_analysis
-   Takes a coh_signal.PowerMorlet object of power data and performs FOOOF power
    analysis.
"""

import numpy as np
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_power import PowerFOOOF, PowerMorlet
import coh_signal


def morlet_analysis(
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
    """
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
        f"{folderpath_processing}\\Settings\\Generic", analysis, ".json"
    )
    morlet_fpath = generate_sessionwise_fpath(
        f"{folderpath_processing}\\Data",
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        f"power-{analysis}",
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)

    ### Data processing
    ## Morlet wavelet power analysis
    morlet = PowerMorlet(signal)
    morlet.process(
        freqs=np.arange(
            analysis_settings["freqs"][0], analysis_settings["freqs"][1] + 1
        ).tolist(),
        n_cycles=analysis_settings["n_cycles"],
        use_fft=analysis_settings["use_fft"],
        decim=analysis_settings["decim"],
        n_jobs=analysis_settings["n_jobs"],
        picks=analysis_settings["picks"],
        zero_mean=analysis_settings["zero_mean"],
        average_windows=analysis_settings["average_windows"],
        average_epochs=analysis_settings["average_epochs"],
        average_timepoints=analysis_settings["average_timepoints"],
        output=analysis_settings["output"],
    )
    if "normalise" in analysis_settings.keys():
        norm_settings = analysis_settings["normalise"]
        morlet.normalise(
            norm_type=norm_settings["norm_type"],
            within_dim=norm_settings["within_dim"],
            exclude_line_noise_window=norm_settings[
                "exclude_line_noise_window"
            ],
            line_noise_freq=norm_settings["line_noise_freq"],
        )
    if save:
        morlet.save_results(fpath=morlet_fpath)

    return morlet


def fooof_analysis(
    signal: PowerMorlet,
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
    """
    PARAMETERS
    ----------
    signal : coh_power.PowerMorlet
    -   The power spcetra to analyse.

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
        f"{folderpath_processing}\\Settings\\Generic", analysis, ".json"
    )
    specific_settings_fpath = generate_sessionwise_fpath(
        f"{folderpath_processing}\\Settings\\Specific",
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        analysis,
        ".json",
    )
    fooof_fpath = generate_sessionwise_fpath(
        f"{folderpath_processing}\\Data",
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        f"power-{analysis}",
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)
    data_settings = load_file(fpath=specific_settings_fpath)

    ### Data processing
    ## FOOOF power analysis
    fooof = PowerFOOOF(signal)
    fooof.process(
        freq_range=analysis_settings["freq_range"],
        peak_width_limits=analysis_settings["peak_width_limits"],
        max_n_peaks=analysis_settings["max_n_peaks"],
        min_peak_height=analysis_settings["min_peak_height"],
        peak_threshold=analysis_settings["peak_threshold"],
        aperiodic_modes=data_settings["aperiodic_modes"],
        show_fit=analysis_settings["show_fit"],
    )
    if save:
        fooof.save_results(fooof_fpath)
