import json

from coh_signal import Signal
from coh_filepath import DataWiseFilepath, AnalysisWiseFilepath, RawFilepath



def preprocessing(
    folderpath_data: str,
    folderpath_extras: str,
    dataset: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str
    ) -> Signal:

    ### Analysis setup
    ## Gets the relevant filepaths
    analysis_settings_fpath = AnalysisWiseFilepath(
        folderpath_extras+'\\settings', analysis, '.json'
    ).path()
    data_settings_fpath = DataWiseFilepath(
        folderpath_extras, dataset, subject, session, task, acquisition, run,
        'settings', '.json'
    ).path()
    raw_fpath = RawFilepath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    ).path()
    annotations_fpath = DataWiseFilepath(
        folderpath_extras, dataset, subject, session, task, acquisition, run,
        'annotations', '.csv'
    ).path()

    ## Loads the analysis settings
    with open(analysis_settings_fpath) as json_file:
        analysis_settings = json.load(json_file)
    with open(data_settings_fpath) as json_file:
        data_settings = json.load(json_file)


    ### Data Preprocessing
    signal = Signal()
    signal.load_raw(raw_fpath)
    signal.load_annotations(annotations_fpath)
    signal.pick_channels(data_settings['ch_names'])
    signal.set_coordinates(
        data_settings['ch_names'], data_settings['ch_coords']
    )
    for key in data_settings['rereferencing'].keys():
        reref_settings = data_settings['rereferencing'][key]
        if key == 'pseudo':
            reref_method = signal.rereference_pseudo
        elif key == 'bipolar':
            reref_method = signal.rereference_bipolar
        elif key == 'CAR':
            reref_method = signal.rereference_CAR
        else:
            raise Exception(
                "Error when rereferencing data:\nThe following rereferencing "
                f"method {key} is not implemented."
            )
        reref_method(
            reref_settings['ch_names_old'], reref_settings['ch_names_new'],
            reref_settings['ch_types_new'], reref_settings['reref_types'],
            reref_settings['ch_coords_new']
        )
    signal.drop_unrereferenced_channels()
    signal.notch_filter(analysis_settings['line_noise'])
    signal.bandpass_filter(
        analysis_settings['lowpass'], analysis_settings['highpass']
    )
    signal.resample(analysis_settings['resample'])
    signal.epoch(analysis_settings['epoch_length'])

    return signal


