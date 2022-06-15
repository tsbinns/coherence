"""Loads and preprocesses ECoG and LFP data stored in the MNE-BIDS format.

METHODS
-------
preprocessing
-   Loads an mne.io.Raw object and preprocesses it in preparation for analysis.
"""

import os
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_raw_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_settings import extract_metadata
from coh_signal import Signal


def preprocessing(
    folderpath_data: str,
    folderpath_preprocessing: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    save: bool = False,
) -> Signal:
    """Loads an mne.io.Raw object, preprocesses it, and epochs it.

    PARAMETERS
    ----------
    folderpath_data : str
    -   The folderpath to the location of the datasets.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing settings and
        derivatives.

    dataset : str
    -   The name of the cohort's raw data and processing data folders.

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

    save : bool; default False
    -   Whether or not to save the preprocessed data

    RETURNS
    -------
    signal : Signal
    -   The preprocessed and epoched data.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    generic_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Generic"
    )
    specific_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Specific"
    )
    analysis_settings_fpath = generate_analysiswise_fpath(
        generic_analysis_folder, analysis, ".json"
    )
    data_settings_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "settings",
        ".json",
    )
    raw_fpath = generate_raw_fpath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    )
    annotations_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "annotations",
        ".csv",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=analysis_settings_fpath)
    data_settings = load_file(fpath=data_settings_fpath)

    ### Data Pre-processing
    signal = Signal()
    signal.raw_from_fpath(raw_fpath)
    if analysis_settings["load_annotations"]:
        signal.load_annotations(annotations_fpath)
    signal.pick_channels(data_settings["ch_names"])
    if data_settings["ch_coords"] is not None:
        signal.set_coordinates(
            data_settings["ch_names"], data_settings["ch_coords"]
        )
    signal.set_regions(
        data_settings["ch_names"],
        data_settings["ch_regions"],
    )
    signal.set_subregions(
        data_settings["ch_names"],
        data_settings["ch_subregions"],
    )
    signal.set_hemispheres(
        data_settings["ch_names"], data_settings["ch_hemispheres"]
    )
    if "combine_channels" in data_settings.keys():
        combine_settings = data_settings["combine_channels"]
        signal.combine_channels(
            ch_names_old=combine_settings["ch_names_old"],
            ch_names_new=combine_settings["ch_names_new"],
            ch_types_new=combine_settings["ch_types_new"],
            ch_coords_new=combine_settings["ch_coords_new"],
            ch_regions_new=combine_settings["ch_regions_new"],
            ch_subregions_new=combine_settings["ch_subregions_new"],
        )
    if analysis_settings["rereference"]:
        for key in data_settings["rereferencing"].keys():
            reref_settings = data_settings["rereferencing"][key]
            if key == "pseudo":
                reref_method = signal.rereference_pseudo
            elif key == "bipolar":
                reref_method = signal.rereference_bipolar
            elif key == "common_average":
                reref_method = signal.rereference_common_average
            else:
                raise Exception(
                    "Error when rereferencing data:\nThe following rereferencing "
                    f"method '{key}' is not implemented."
                )
            reref_method(
                ch_names_old=reref_settings["ch_names_old"],
                ch_names_new=reref_settings["ch_names_new"],
                ch_types_new=reref_settings["ch_types_new"],
                ch_reref_types=reref_settings["ch_reref_types"],
                ch_coords_new=reref_settings["ch_coords_new"],
                ch_regions_new=reref_settings["ch_regions_new"],
                ch_subregions_new=reref_settings["ch_subregions_new"],
                ch_hemispheres_new=reref_settings["ch_hemispheres_new"],
            )
        signal.drop_unrereferenced_channels()
        signal.order_channels(data_settings["post_reref_organisation"])
    if analysis_settings["line_noise"] is not None:
        signal.notch_filter(analysis_settings["line_noise"])
    if analysis_settings["bandpass"] is not None:
        signal.bandpass_filter(
            analysis_settings["bandpass"][1], analysis_settings["bandpass"][0]
        )
    if analysis_settings["window_length"] is not None:
        signal.window(analysis_settings["window_length"])
    if analysis_settings["epoch_length"] is not None:
        signal.epoch(analysis_settings["epoch_length"])
    if analysis_settings["resample"] is not None:
        signal.resample(analysis_settings["resample"])
    if analysis_settings["n_shuffles"] > 0:
        signal.shuffle(
            channels=data_settings["shuffle_channels"],
            n_shuffles=analysis_settings["n_shuffles"],
            rng_seed=analysis_settings["rng_seed"],
        )

    ## Adds metadata about the preprocessed data
    metadata = extract_metadata(settings=data_settings)
    signal.add_metadata(metadata)

    if save:
        preprocessed_data_folder = os.path.join(
            folderpath_preprocessing, "Data"
        )
        preprocessed_data_fpath = generate_sessionwise_fpath(
            preprocessed_data_folder,
            dataset,
            subject,
            session,
            task,
            acquisition,
            run,
            f"preprocessed-{analysis}",
            ".json",
        )
        signal.save_as_dict(fpath=preprocessed_data_fpath)

    return signal


def preprocessing_for_annotations(
    folderpath_data: str,
    folderpath_preprocessing: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
) -> Signal:
    """Loads an mne.io.Raw object and preprocesses it for annotating

    PARAMETERS
    ----------
    folderpath_data : str
    -   The folderpath to the location of the datasets.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing settings and
        derivatives.

    dataset : str
    -   The name of the cohort's raw data and processing data folders.

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

    RETURNS
    -------
    signal : Signal
    -   The preprocessed and epoched data.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    generic_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Generic"
    )
    specific_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Specific"
    )
    analysis_settings_fpath = generate_analysiswise_fpath(
        generic_analysis_folder, analysis, ".json"
    )
    data_settings_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "settings",
        ".json",
    )
    raw_fpath = generate_raw_fpath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=analysis_settings_fpath)
    data_settings = load_file(fpath=data_settings_fpath)

    ### Data Pre-processing
    signal = Signal()
    signal.raw_from_fpath(raw_fpath)
    signal.pick_channels(data_settings["ch_names"])
    if data_settings["ch_coords"] is not None:
        signal.set_coordinates(
            data_settings["ch_names"], data_settings["ch_coords"]
        )
    signal.set_regions(data_settings["ch_names"], data_settings["ch_regions"])
    signal.set_hemispheres(
        data_settings["ch_names"], data_settings["ch_hemispheres"]
    )
    if "combine_channels" in data_settings.keys():
        combine_settings = data_settings["combine_channels"]
        signal.combine_channels(
            ch_names_old=combine_settings["ch_names_old"],
            ch_names_new=combine_settings["ch_names_new"],
            ch_types_new=combine_settings["ch_types_new"],
            ch_coords_new=combine_settings["ch_coords_new"],
            ch_regions_new=combine_settings["ch_regions_new"],
        )
    if analysis_settings["rereference"]:
        for key in data_settings["rereferencing"].keys():
            reref_settings = data_settings["rereferencing"][key]
            if key == "pseudo":
                reref_method = signal.rereference_pseudo
            elif key == "bipolar":
                reref_method = signal.rereference_bipolar
            elif key == "common_average":
                reref_method = signal.rereference_common_average
            else:
                raise Exception(
                    "Error when rereferencing data:\nThe following rereferencing "
                    f"method '{key}' is not implemented."
                )
            reref_method(
                ch_names_old=reref_settings["ch_names_old"],
                ch_names_new=reref_settings["ch_names_new"],
                ch_types_new=reref_settings["ch_types_new"],
                ch_reref_types=reref_settings["ch_reref_types"],
                ch_coords_new=reref_settings["ch_coords_new"],
                ch_regions_new=reref_settings["ch_regions_new"],
                ch_subregions_new=reref_settings["ch_subregions_new"],
                ch_hemispheres_new=reref_settings["ch_hemispheres_new"],
            )
        signal.order_channels(data_settings["post_reref_organisation"])
    if analysis_settings["line_noise"] is not None:
        signal.notch_filter(analysis_settings["line_noise"])
    if analysis_settings["bandpass"] is not None:
        signal.bandpass_filter(
            analysis_settings["bandpass"][1], analysis_settings["bandpass"][0]
        )
    if analysis_settings["resample"] is not None:
        signal.resample(analysis_settings["resample"])

    ## Adds metadata about the preprocessed data
    metadata = extract_metadata(settings=data_settings)
    signal.add_metadata(metadata)

    return signal
