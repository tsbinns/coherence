import mne
import numpy as np



def annotations_replace_dc(raw):
    """ Sets the 'DC Corrections' annotation to be 'BAD'

    PARAMETERS
    ----------
    raw : MNE Raw object
        The data whose annotations are to be changed.

    RETURNS
    ----------
    raw : MNE Raw object
        The data whose annotations have been changed.
    
    """

    raw.annotations.duration[raw.annotations.description == 'DC Correction/'] = 6
    raw.annotations.onset[raw.annotations.description == 'DC Correction/'] = raw.annotations.onset[
        raw.annotations.description == 'DC Correction/'] - 2
    raw.annotations.description[raw.annotations.description == 'DC Correction/'] = 'BAD'


    return raw



def crop_artefacts(raw):
    """ Removes segments labelled as artefacts from the data in lieu of a way to perform this in MNE.
    
    PARAMETERS
    ----------
    raw : MNE Raw object
        The data whose artefact segments are to be removed.

    RETURNS
    ----------
    raw : MNE Raw object
        The data whose artefact segments have been removed.
    """

    # Marks the 'DC Correction' annotation as 'BAD' and extracts the 'BAD' annotations
    raw = annotations_replace_dc(raw)
    bad_idc = np.flatnonzero(np.char.startswith(raw.annotations.description, 'BAD'))
    bad = raw.annotations[bad_idc]

    # Cuts the 'BAD' segments from the data
    if bad:
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
        cropped_raw.annotations.delete(bad_idc)
    else:
        cropped_raw = raw


    return cropped_raw



def epoch_data(raw, epoch_len, include_shuffled=True):
    """ Epochs the data.

    PARAMETERS
    ----------
    raw : MNE Raw object
        The data to be epoched.
    epoch_len : int | float
        The duration (in seconds) of segments to epoch the data into.
    include_shuffled : bool
        States whether or not new channels of LFP data should be generated in which the order of epochs is shuffled
        randomly. When the coherence of this shuffled data is compared to genuine data, a baseline coherence is produced
        against which genuine value can be compared. If True (default), shuffled data is generated.
    
    RETURNS
    ----------
    epoched : MNE Epoch object
        The epoched data.    
    """

    # Epochs data
    epoched = mne.make_fixed_length_epochs(raw, epoch_len)

    # Optionally adds shuffled data
    if include_shuffled:

        epoched.load_data()
        epoched_shuffled = epoched.copy()
        epoched_shuffled.pick_types(dbs=True)
        ch_name = epoched_shuffled.ch_names[0]
        
        # Randomly shuffles all epochs x times to create x shuffled channels
        np.random.seed(seed=0)
        n_shuffles = 10
        shuffled_order = []
        shuffled_epochs = []
        for i in range(n_shuffles):
            shuffled_order = np.arange(0,len(epoched_shuffled.events))
            np.random.shuffle(shuffled_order)
            shuffled_epochs.append(mne.concatenate_epochs([epoched_shuffled[shuffled_order]]))
            shuffled_epochs[i].rename_channels({ch_name: 'SHUFFLED-'+str(i)+'_'+ch_name})
        
        epoched.add_channels(shuffled_epochs)

    
    return epoched



def process(raw, epoch_len, annotations=None, channels=None, resample=None, highpass=None, lowpass=None, notch=None,
            include_shuffled=True, verbose=True):
    """ Preprocessing of the raw data in preparation for coherence analysis.
    
    PARAMETERS
    ----------
    raw : MNE Raw object
        The data to be processed.
    epoch_len : int | float
        The duration (in seconds) of segments to epoch the data into.
    annotations : MNE Annotations object | None (default)
        The annotations to eliminate BAD segments from the data.
    channels : list of str | None (default)
        The names of the channels to pick from the data.
    resample : int | float | None (default)
        The frequency (in Hz) at which to resample the data.
    highpass : int | float | None (default)
        The frequency (in Hz) at which to highpass filter the data.
    lowpass : int | float | None (default)
        The frequency (in Hz) at which to lowpass filter the data.
    notch : list of int | list of float | None (default)
        The frequencies (in Hz) at which to notch filter the data.
    include_shuffled : bool
        States whether or not new channels of LFP data should be generated in which the order of epochs is shuffled
        randomly. When the coherence of this shuffled data is compared to genuine data, a baseline coherence is produced
        against which genuine value can be compared. If True (default), shuffled data is generated.
    verbose : bool
        States whether or not updates on the various processing stages should be given. If True (default), these updates
        are given.

    RETURNS
    ----------
    epoched : MNE Epoch object
        The processed, epoched data.

    """

    # Adds annotations and removes artefacts
    if annotations != None:
        no_annots = raw._annotations
        raw.set_annotations(annotations)
        #raw = crop_artefacts(raw)
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
            new_types[name] = 'dbs'
    raw.set_channel_types(new_types)

    # Rereferencing
    raw.load_data()
    raw.set_eeg_reference(ch_type='ecog')
    new_names = {}
    for name in channels:
        if name[:4] == 'ECOG':
            new_names[name] = name + '_CAR'
    raw.rename_channels(new_names)
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

