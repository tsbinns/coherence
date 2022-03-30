"""Performs power analysis on pre-processed data.

METHODS
-------
power_morlet_analysis
-   Takes a coh_signal.Signal object of pre-processed data and performs Morlet
    wavelet power analysis.

power_FOOOF_analysis
-   Takes a coh_signal.PowerMorlet object of power data and performs FOOOF power
    nalysis.
"""

import json
import numpy as np
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
)
from coh_power import PowerFOOOF, PowerMorlet
import coh_signal


def power_morlet_analysis(
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
    morlet_fpath = generate_sessionwise_fpath(
        folderpath_extras,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "power-morlet",
        ".json",
    )

    ## Loads the analysis settings
    with open(analysis_settings_fpath, encoding="utf-8") as file:
        analysis_settings = json.load(file)
    morlet_settings = analysis_settings["power_morlet"]
    power_norm_settings = morlet_settings["normalise_power"]
    itc_norm_settings = morlet_settings["normalise_itc"]

    ### Data processing
    ## Morlet wavelet power analysis
    morlet = PowerMorlet(signal)
    morlet.process(
        freqs=np.arange(
            morlet_settings["freqs"][0], morlet_settings["freqs"][1] + 1
        ).tolist(),
        n_cycles=morlet_settings["n_cycles"],
        use_fft=morlet_settings["use_fft"],
        return_itc=morlet_settings["return_itc"],
        decim=morlet_settings["decim"],
        n_jobs=morlet_settings["n_jobs"],
        picks=morlet_settings["picks"],
        zero_mean=morlet_settings["zero_mean"],
        average_epochs=morlet_settings["average_epochs"],
        average_timepoints_power=morlet_settings["average_timepoints_power"],
        average_timepoints_itc=morlet_settings["average_timepoints_itc"],
        output=morlet_settings["output"],
    )
    if power_norm_settings["apply"]:
        morlet.normalise(
            norm_type=power_norm_settings["norm_type"],
            apply_to="power",
            within_dim=power_norm_settings["within_dim"],
            exclude_line_noise_window=power_norm_settings[
                "exclude_line_noise_window"
            ],
        )
    if itc_norm_settings["apply"]:
        morlet.normalise(
            norm_type=itc_norm_settings["norm_type"],
            apply_to="itc",
            within_dim=itc_norm_settings["within_dim"],
            exclude_line_noise_window=itc_norm_settings[
                "exclude_line_noise_window"
            ],
        )
    if save:
        morlet.save_results(morlet_fpath)

    return morlet


def power_fooof_analysis(
    signal: PowerMorlet,
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
    signal : coh_power.PowerMorlet
    -   The power spcetra to analyse.

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
    fooof_fpath = generate_sessionwise_fpath(
        folderpath_extras,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "power-FOOOF",
        ".json",
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
    with open(analysis_settings_fpath, encoding="utf-8") as file:
        analysis_settings = json.load(file)
    fooof_settings = analysis_settings["power_FOOOF"]
    if fooof_settings["max_n_peaks"] == "infinity":
        fooof_settings["max_n_peaks"] = float("inf")
    with open(data_settings_fpath, encoding="utf-8") as json_file:
        data_settings = json.load(json_file)

    ### Data processing
    ## FOOOF power analysis
    fooof = PowerFOOOF(signal)
    fooof.process(
        freq_range=fooof_settings["freq_range"],
        peak_width_limits=fooof_settings["peak_width_limits"],
        max_n_peaks=fooof_settings["max_n_peaks"],
        min_peak_height=fooof_settings["min_peak_height"],
        peak_threshold=fooof_settings["peak_threshold"],
        aperiodic_modes=data_settings["power_FOOOF"]["aperiodic_modes"],
        show_fit=fooof_settings["show_fit"],
    )
    if save:
        fooof.save_results(fooof_fpath)
