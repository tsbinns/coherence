import mne
import numpy as np
from copy import deepcopy



def annotations_replace_dc(raw):
    """ Sets the 'DC Corrections' annotation to be 'BAD'

    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data whose annotations are to be changed.


    RETURNS
    ----------
    raw : MNE Raw object
    -   The data whose annotations have been changed.
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
    -   The data whose artefact segments are to be removed.


    RETURNS
    ----------
    raw : MNE Raw object
    -   The data whose artefact segments have been removed.
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
            print(f'Removing {bad.onset[n]} - {new_start}')
            if new_start > start:
                start = new_start

            if stop > raw._last_time:
                stop = raw._last_time
            if 0 < start < stop <= raw._last_time:
                cropped_raw = mne.concatenate_raws([cropped_raw, raw.copy().crop(start, stop)])
        cropped_raw.annotations.delete(bad_idc)
    else:
        cropped_raw = raw


    return cropped_raw



def epoch_data(raw, epoch_len, include_shuffled=True):
    """ Epochs the data.

    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data to be epoched.

    epoch_len : int | float
    -   The duration (in seconds) of segments to epoch the data into.

    include_shuffled : bool
    -   States whether or not new channels of LFP data should be generated in which the order of epochs is shuffled
        randomly. When the coherence of this shuffled data is compared to genuine data, a baseline coherence is produced
        against which genuine value can be compared. If True (default), shuffled data is generated.
    

    RETURNS
    ----------
    epoched : MNE Epoch object
    -   The epoched data.    
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



def process(raw, epoch_len, annotations=None, channels=None, rereferencing=None, resample=None, highpass=None,
            lowpass=None, notch=None, include_shuffled=True, verbose=True):
    """ Preprocessing of the raw data in preparation for coherence analysis.
    
    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data to be processed.

    epoch_len : int | float
    -   The duration (in seconds) of segments to epoch the data into.

    annotations : MNE Annotations object | None (default)
    -   The annotations to eliminate BAD segments from the data.

    channels : list of str | None (default)
    -   The names of the channels to pick from the data.

    rereferencing : dict
    -   A dictionary containing instructions for rereferencing. The keys of the dictionary can be 'CAR' (common
        average referencing) or 'bipolar' (bipolar referencing). The values of these keys should be the channels to
        rereference using these referencing types.
    -   For 'CAR', the values should be the type of channels to rereference (e.g. 'ecog', 'dbs').
    -   For 'bipolar', the value should be a list of sublists, with sublists of shape (n_rereferenced_channels x 3). The
        first entry of each sublist should be the name of a channel to use as the anode of the bipolar rereference, the
        second entry should be the name of a channel to use as the cathode, and the third entry should be the name of
        the new channel created by this rereferencing.

    resample : int | float | None (default)
    -   The frequency (in Hz) at which to resample the data.

    highpass : int | float | None (default)
    -   The frequency (in Hz) at which to highpass filter the data.

    lowpass : int | float | None (default)
    -   The frequency (in Hz) at which to lowpass filter the data.

    notch : list of int | list of float | None (default)
    -   The frequencies (in Hz) at which to notch filter the data.

    include_shuffled : bool
    -   States whether or not new channels of LFP data should be generated in which the order of epochs is shuffled
        randomly. When the coherence of this shuffled data is compared to genuine data, a baseline coherence is produced
        against which genuine value can be compared. If True (default), shuffled data is generated.

    verbose : bool
    -   States whether or not updates on the various processing stages should be given. If True (default), these updates
        are given.


    RETURNS
    ----------
    epoched : MNE Epoch object
    -   The processed, epoched data.
    """

    ## Sets up preprocessing
    # Selects channels
    if channels != None:
        if verbose:
            print("Picking specified channels")
        raw.pick_channels(channels)

    # Adds annotations and removes artefacts
    if annotations != None:
        if verbose:
            print("Setting annotations and removing artefacts")
        raw.set_annotations(annotations)
    
    # Gets data from the Raw object
    channels = raw.info.ch_names.copy()
    raw.load_data()
    raw_data = raw.get_data(reject_by_annotation='omit').copy()


    ## Rereferencing
    if rereferencing != None:
        if verbose:
            print("Rereferencing the data")

        reref_data = [] # holder for rereferenced data
        new_channs = [] # holder for the names of the new channels of rereferenced data
        reref_type = [] # holder for the type of rereferencing (e.g. bipolar, CAR) of each channel
        channs_type = [] # holder for the type of data in channels (e.g. ecog, dbs)

        # Rereferences data
        ref_types = ['none', 'bipolar', 'CAR'] # types of rereferencing that are supported
        n_refs = 0 # used to check that the number of rereferencing methods applied is the same as those requested
        for ref_type in ref_types:
            if ref_type in rereferencing.keys():

                if ref_type == 'none': # leaves the data as is
                    for chann_i in range(len(rereferencing['none']['old'])): # for each channel to leave untouched
                        old_name = rereferencing['none']['old'][chann_i]
                        new_name = rereferencing['none']['new'][chann_i]
                        new_channs.append(new_name)
                        reref_data.append(raw_data[channels.index(old_name)]) # adds the original data,...
                        reref_type.append(rereferencing['none']['ref_type'][chann_i]) #... the desired type of...
                        #... rereferencing to set,...
                        channs_type.append(rereferencing['none']['chann_type'][chann_i]) #... and the type of the...
                        #... data to their respective holders
                
                if ref_type == 'bipolar': # bipolar rereferences data
                    for chann_i in range(len(rereferencing['bipolar']['old'])): # for each set of two channels to reref.
                        old_names = rereferencing['bipolar']['old'][chann_i] # names of the channels to bipolar reref.
                        new_name = rereferencing['bipolar']['new'][chann_i] # name of the new channel to create
                        new_channs.append(new_name)
                        reref_data.append(raw_data[channels.index(old_names[0])] -
                                          raw_data[channels.index(old_names[1])]) # anode - cathode => bipolar reref.
                        reref_type.append('bipolar')
                        channs_type.append(rereferencing['bipolar']['chann_type'][chann_i])

                if ref_type == 'CAR': # common average rereferences data
                    for channs_i in range(len(rereferencing['CAR']['old'])): # for each group of channels to average
                        old_names = rereferencing['CAR']['old'][channs_i] # names of the original channels
                        new_names = rereferencing['CAR']['new'][channs_i] # new names for the averaged data
                        avg_data = raw_data[[i for i, x in enumerate(channels) if x in old_names]].mean(axis=0) # the...
                        #... average of the data in these channels
                        for chann_i in range(len(new_names)): # for each of the original channels
                            old_name = old_names[chann_i]
                            new_name = new_names[chann_i]
                            new_channs.append(new_name)
                            reref_data.append(raw_data[channels.index(old_name)] - avg_data) # original - average...
                            #... => CAR reref.
                            reref_type.append('CAR')
                            channs_type.append(rereferencing['CAR']['chann_type'][channs_i][chann_i])

                n_refs += 1
        if n_refs != len(rereferencing.keys()): # makes sure the proper number of rereferencing methods have been used
            raise ValueError(f'{len(rereferencing.keys())} forms of rereferencing were requested, but {n_refs} was/were applied. The accepted rereferencing types are CAR and bipolar.')

        # Sorts the data together based on rereferencing type
        channs_i_sorted = [] # holder for the indices of the sorted data
        for ref_type in np.unique(reref_type): # for each type of reref. applied
            ref_channs_i = [i for i, x in enumerate(reref_type) if x == ref_type] # finds the indices of these channels
            ref_channs = [new_channs[i] for i in ref_channs_i] # gets the name of these channels
            idx = range(len(ref_channs_i))
            sorted_i = sorted(idx, key=lambda x:ref_channs[x]) # sorts the channels alphabetically
            channs_i_sorted.extend([ref_channs_i[i] for i in sorted_i]) # gets the indices of the sorted channels in...
            #... the original data
        new_channs_sorted = [new_channs[i] for i in channs_i_sorted] # sorts the names of the channels
        reref_data_sorted = [reref_data[i] for i in channs_i_sorted] # sorts the data itself
        channs_type_sorted = [channs_type[i] for i in channs_i_sorted] # sorts the type of the data

        # Makes a new Raw object based on the rereferenced data
        raw_info = mne.create_info(ch_names=new_channs_sorted, sfreq=raw.info['sfreq'], ch_types=channs_type_sorted)
        raw = mne.io.RawArray(data=reref_data_sorted, info=raw_info)


    ## Filtering and resampling
    # Notch filters data
    if notch.all() != None:
        if verbose:
            print("Notch filtering the data")
        raw.notch_filter(notch)
    
    # Bandpass filters data
    if highpass != None or lowpass != None:
        if verbose:
            print("Bandpass filtering the data")
        raw.filter(highpass, lowpass)

    # Resamples data
    if resample != None:
        if verbose:
            print("Resampling the data")
        raw.resample(resample)


    # Epochs data (and adds shuffled LFP data if requested)
    if epoch_len != None:
        if verbose:
            print("Epoching data")
        epoched = epoch_data(raw, epoch_len, include_shuffled)


    return epoched

