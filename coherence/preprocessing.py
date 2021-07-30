import mne
import numpy as np


def annotations_get_bad(annot):
    i = np.flatnonzero(np.char.startswith(annot.description, 'BAD'))
    return annot[i], i


def annotations_replace_dc(raw):
    raw.annotations.duration[raw.annotations.description == 'DC Correction/'] = 6
    raw.annotations.onset[raw.annotations.description == 'DC Correction/'] = raw.annotations.onset[
                                                                                 raw.annotations.description == 'DC Correction/'] - 2
    raw.annotations.description[raw.annotations.description == 'DC Correction/'] = 'BAD'
    return raw


def crop_artefacts(raw):
    raw = annotations_replace_dc(raw)
    bad, bad_idx = annotations_get_bad(raw.annotations)
    start = 0
    stop = bad.onset[0]
    cropped_raw = raw.copy().crop(start, stop)
    for n, a in enumerate(bad, 0):
        if n + 1 < len(bad):
            stop = bad.onset[n + 1]
        else:
            stop = raw._last_time

        new_start = bad.onset[n] + bad.duration[n]
        if new_start > start:
            start = new_start

        if stop > raw._last_time:
            stop = raw._last_time
        if 0 < start < stop < raw._last_time:
            cropped_raw = mne.concatenate_raws([cropped_raw, raw.copy().crop(start, stop)])
            print((start, stop))
    cropped_raw.annotations = cropped_raw.annotations.pop(bad_idx)

    return cropped_raw


def epoch_data(raw, epoch_len, include_shuffled=False):
    epoched = mne.make_fixed_length_epochs(raw, epoch_len)
    epoched.load_data()

    if include_shuffled:
        epoched_shuffled = epoched.copy()
        epoched_shuffled.pick_types(seeg=True)
        ch_name = epoched_shuffled.ch_names[0]
        epoched_shuffled.rename_channels({ch_name: 'SHUFFLED_'+ch_name})
        shuffled_order = [len(epoched_shuffled.events)-1,
                          *np.arange(0,len(epoched_shuffled.events)-1)]
        epoched_shuffled = mne.concatenate_epochs([epoched_shuffled[shuffled_order]])
    
    return epoched.add_channels([epoched_shuffled])


def process(raw, annotations=None, channels=None, resample=None, highpass=None,
            lowpass=None, notch=None, epoch_len=None, include_shuffled=False,
            verbose=True):
    # Adds annotations and removes artefacts
    if annotations != None:
        no_annots = raw._annotations
        raw.set_annotations(annotations)
        raw = crop_artefacts(raw)
        #raw._annotations = no_annots
        if verbose:
            print("Setting annotations and removing artefacts")

    # Selects channels
    if channels != None:
        raw.pick_channels(channels)
        if verbose:
            print("Picking specified channels")

    # Sets LFP channel types to type SEEG
    names = raw.info.ch_names
    new_types = {}
    for name in names:
        if name[:3] == 'LFP':
            new_types[name] = 'seeg'
    raw.set_channel_types(new_types)

    # Rereferencing
    raw.load_data()
    raw.set_eeg_reference(ch_type='ecog')
    raw = mne.set_bipolar_reference(raw, 'LFP_L_1_STN_BS', 'LFP_L_8_STN_BS', ch_name='LFP_L_18_STN_BS')
    if verbose:
        print("Rereferencing the data")

    # Notch filters data
    if notch.all() != None:
        raw.notch_filter(notch)
        if verbose:
            print("Notch filtering the data")
    
    # Bandpass filters data
    if highpass != None or lowpass != None:
        """
        if bandpass[1] >= raw.info['sfreq']/2:
            bandpass[1] = raw.info['sfreq']/2-1
            if verbose:
                print("Reducing the upper limit of the bandpass frequency to ", bandpass[1], "Hz")
        """
        raw.filter(highpass, lowpass)
        if verbose:
            print("Bandpass filtering the data")

    # Resamples data
    if resample != None:
        raw.resample(resample)
        if verbose:
            print("Resampling the data")

    # Epochs data (and adds shuffled LFP data if requested)
    if epoch_len != None:
        epoched = epoch_data(raw, epoch_len, include_shuffled)
        if verbose:
            print("Epoching data")

    return epoched

