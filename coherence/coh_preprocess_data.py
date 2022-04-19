"""Loads and preprocesses ECoG and LFP data stored in the MNE-BIDS format.

METHODS
-------
preprocessing
-   Loads an mne.io.Raw object and preprocesses it in preparation for analysis.
"""

from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_raw_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_settings import ExtractMetadata
from coh_signal import Signal


def preprocessing(
    folderpath_data: str,
    folderpath_extras: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
) -> Signal:
    """Loads an mne.io.Raw object, preprocesses it, and epochs it.

    PARAMETERS
    ----------
    folderpath_data : str
    -   The folderpath to the location of the datasets.

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

    RETURNS
    -------
    signal : Signal
    -   The preprocessed and epoched data.
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
    raw_fpath = generate_raw_fpath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    )
    annotations_fpath = generate_sessionwise_fpath(
        folderpath_extras,
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
    analysis_settings = analysis_settings["preprocessing"]
    data_settings = load_file(fpath=data_settings_fpath)

    ### Data Pre-processing
    signal = Signal()
    signal.load_raw(raw_fpath)
    if analysis_settings["load_annotations"]:
        signal.load_annotations(annotations_fpath)
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
                reref_settings["ch_names_old"],
                reref_settings["ch_names_new"],
                reref_settings["ch_types_new"],
                reref_settings["reref_types"],
                reref_settings["ch_coords_new"],
                reref_settings["ch_regions_new"],
                reref_settings["ch_hemispheres_new"],
            )
        signal.drop_unrereferenced_channels()
        signal.order_channels(data_settings["post_reref_organisation"])
    if "line_noise" in analysis_settings.keys():
        signal.notch_filter(analysis_settings["line_noise"])
    if (
        "lowpass" in analysis_settings.keys()
        and "highpass" in analysis_settings.keys()
    ):
        signal.bandpass_filter(
            analysis_settings["lowpass"], analysis_settings["highpass"]
        )
    if "epoch_length" in analysis_settings.keys():
        signal.epoch(analysis_settings["epoch_length"])
    if "resample" in analysis_settings.keys():
        signal.resample(analysis_settings["resample"])

    ## Adds metadata about the preprocessed data
    metadata = ExtractMetadata(data_settings).metadata
    signal.add_metadata(metadata)

    return signal
