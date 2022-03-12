"""Loads and preprocesses ECoG and LFP data stored in the MNE-BIDS format.

METHODS
-------
preprocessing
-   Loads an mne.io.Raw object, preprocesses it, and epochs it.
"""


import json

from coh_filepath import SessionwiseFilepath, AnalysiswiseFilepath, RawFilepath
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
    analysis_settings_fpath = AnalysiswiseFilepath(
        folderpath_extras + "\\settings", analysis, ".json"
    ).path()
    data_settings_fpath = SessionwiseFilepath(
        folderpath_extras,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "settings",
        ".json",
    ).path()
    raw_fpath = RawFilepath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    ).path()
    annotations_fpath = SessionwiseFilepath(
        folderpath_extras,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "annotations",
        ".csv",
    ).path()

    ## Loads the analysis settings
    with open(analysis_settings_fpath, encoding="utf-8") as json_file:
        analysis_settings = json.load(json_file)
        analysis_settings = analysis_settings["preprocessing"]
    with open(data_settings_fpath, encoding="utf-8") as json_file:
        data_settings = json.load(json_file)

    ### Data Pre-processing
    signal = Signal()
    signal.load_raw(raw_fpath)
    signal.load_annotations(annotations_fpath)
    signal.pick_channels(data_settings["ch_names"])
    signal.set_coordinates(
        data_settings["ch_names"], data_settings["ch_coords"]
    )
    signal.set_regions(data_settings["ch_names"], data_settings["ch_regions"])
    for key in data_settings["rereferencing"].keys():
        reref_settings = data_settings["rereferencing"][key]
        if key == "pseudo":
            reref_method = signal.rereference_pseudo
        elif key == "bipolar":
            reref_method = signal.rereference_bipolar
        elif key == "CAR":
            reref_method = signal.rereference_CAR
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
        )
    signal.drop_unrereferenced_channels()
    signal.notch_filter(analysis_settings["line_noise"])
    signal.bandpass_filter(
        analysis_settings["lowpass"], analysis_settings["highpass"]
    )
    signal.resample(analysis_settings["resample"])
    signal.epoch(analysis_settings["epoch_length"])
    signal.order_channels(data_settings["post_reref_organisation"])

    ## Adds metadata about the preprocessed data
    metadata = ExtractMetadata(data_settings).metadata
    signal.add_metadata(metadata)

    return signal
