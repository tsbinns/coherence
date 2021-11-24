import mne
import numpy as np
from copy import deepcopy

from numpy.core.fromnumeric import repeat



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



def reref_data(data, info, rereferencing, ch_coords):
    """ Rereferences the data.

    PARAMETERS
    ----------
    data : array of shape n_channels x n_datapoints
    -   The data to analyse, obtained from Raw.get_data().

    info : MNE Raw object information dict
    -   The information dict from the MNE Raw object.

    rereferencing : dict
    -   A dictionary containing instructions for rereferencing. The keys of the dictionary can be 'CAR' (common
        average referencing) or 'bipolar' (bipolar referencing). The values of these keys should be the channels to
        rereference using these referencing types.
    -   For 'CAR', the values should be the type of channels to rereference (e.g. 'ecog', 'dbs').
    -   For 'bipolar', the value should be a list of sublists, with sublists of shape (n_rereferenced_channels x 3). The
        first entry of each sublist should be the name of a channel to use as the anode of the bipolar rereference, the
        second entry should be the name of a channel to use as the cathode, and the third entry should be the name of
        the new channel created by this rereferencing.

    ch_coords : list of lists
    -   The coordinates of the channel positions in the Raw object.

    RETURNS
    ----------
    raw : MNE Raw object
    -   A Raw object created from the rereferenced data.

    extra_info : dict
    -   A dictionary containing additional information following rereferencing (this data cannot be added to raw.info as
        MNE does not allow this).
    """

    channels = info.ch_names.copy()

    reref_data = [] # holder for rereferenced data
    new_channs = [] # holder for the names of the new channels of rereferenced data
    reref_type = [] # holder for the type of rereferencing (e.g. bipolar, CAR) of each channel
    channs_type = [] # holder for the type of data in channels (e.g. ecog, dbs)
    new_coords = [] # holder for the coordinates of the new channels

    # Rereferences data
    ref_types = ['none', 'bipolar', 'CAR'] # types of rereferencing that are supported
    n_refs = 0 # used to check that the number of rereferencing methods applied is the same as those requested
    for ref_type in ref_types:
        if ref_type in rereferencing.keys():

            if ref_type == 'none': # leaves the data as is
                reref = rereferencing['none']
                for chann_i in range(len(reref['old'])): # for each channel to leave untouched
                    old_name = reref['old'][chann_i]
                    new_name = reref['new'][chann_i]
                    new_channs.append(new_name)
                    reref_data.append(data[channels.index(old_name)]) # adds the original data,...
                    reref_type.append(reref['ref_type'][chann_i]) #... the desired type of...
                    #... rereferencing to set,...
                    channs_type.append(reref['chann_type'][chann_i]) #... the type of the data,
                    # and the coordinates of the channel to their respective holders
                    if reref['coords'] == []: # if no coordinates are specified,...
                        new_coords.append(ch_coords[channels.index(old_name)]) #... use those from the original channels
                    else: # if coordinates are specified,...
                        if len(reref['coords']) != len(reref['new']): #... check they are the correct format,
                            raise ValueError(f"There is a mismatch between channels ({len(reref['new'])}) and coordinates ({len(reref['coords'])}).")
                        new_coords.append(reref['coords'][reref['new'].index(new_name)]) #... then add the coordinates

            if ref_type == 'bipolar': # bipolar rereferences data
                reref = rereferencing['bipolar']
                for chann_i in range(len(reref['old'])): # for each set of two channels to reref.
                    if len(reref['old'][chann_i]) > 2:
                        raise ValueError(f"Only 2 channels should be used for bipolar referencing, but {len(reref['old'][chann_i])} are requested.")
                    old_names = reref['old'][chann_i] # names of the channels to bipolar reref.
                    new_name = reref['new'][chann_i] # name of the new channel to create
                    new_channs.append(new_name)
                    reref_data.append(data[channels.index(old_names[0])] -
                                        data[channels.index(old_names[1])]) # anode - cathode => bipolar reref.
                    reref_type.append('bipolar')
                    channs_type.append(reref['chann_type'][chann_i])
                    if reref['coords'] == []: # if no coordinates are specified,...
                        #... calculate the new coordinates based on the original channel coordinates
                        new_coords.append(np.mean([ch_coords[channels.index(old_names[0])],
                                                   ch_coords[channels.index(old_names[1])]], axis=0).tolist())
                    else: # if coordinates are specified,...
                        if len(reref['coords']) != len(reref['new']): #... check they are the correct format,
                            raise ValueError(f"There is a mismatch between channels ({len(reref['new'])}) and coordinates ({len(reref['coords'])}).")
                        new_coords.append(reref['coords'][reref['new'].index(new_name)]) #... then add the coordinates
                        
            if ref_type == 'CAR': # common average rereferences data
                reref = rereferencing['CAR']
                for channs_i in range(len(reref['old'])): # for each group of channels to average
                    old_names = reref['old'][channs_i] # names of the original channels
                    new_names = reref['new'][channs_i] # new names for the averaged data
                    avg_data = data[[i for i, x in enumerate(channels) if x in old_names]].mean(axis=0) # the...
                    #... average of the data in these channels
                    for chann_i in range(len(new_names)): # for each of the original channels
                        old_name = old_names[chann_i]
                        new_name = new_names[chann_i]
                        new_channs.append(new_name)
                        reref_data.append(data[channels.index(old_name)] - avg_data) # original - average...
                        #... => CAR reref.
                        reref_type.append('CAR')
                        channs_type.append(reref['chann_type'][channs_i][chann_i])
                        if reref['coords'] == []: # if no coordinates are specified,...
                            new_coords.append(ch_coords[channels.index(old_name)]) #... use those from the original channels
                        else: # if coordinates are specified,...
                            if len(reref['coords'][channs_i]) != len(new_names): #... check they are the correct format,
                                raise ValueError(f"There is a mismatch between channels ({len(new_names)}) and coordinates ({len(reref['coords'][channs_i])}).")
                            new_coords.append(reref['coords'][channs_i][reref['new'][channs_i].index(new_name)]) #... then add the coordinates

            n_refs += 1
    if n_refs != len(rereferencing.keys()): # makes sure the proper number of rereferencing methods have been used
        raise ValueError(f'{len(rereferencing.keys())} forms of rereferencing were requested, but {n_refs} was/were applied. The accepted rereferencing types are CAR and bipolar.')

    # Sorts the data together based on rereferencing type
    channs_i_sorted = [] # holder for the indices of the sorted data
    for ref_type in np.unique(reref_type): # for each type of reref. applied
        ref_channs_i = [i for i, x in enumerate(reref_type) if x == ref_type] # finds the indices of these channels
        """ OLD IMPLEMENTATION THAT BREAKS WHEN COMPARING '1_' AND '10_'
        ref_channs = [new_channs[i] for i in ref_channs_i] # gets the name of these channels
        idx = range(len(ref_channs_i))
        sorted_i = sorted(idx, key=lambda x:ref_channs[x]) # sorts the channels alphabetically
        channs_i_sorted.extend([ref_channs_i[i] for i in sorted_i]) # gets the indices of the sorted channels in...
        #... the original data
        """
        channs_i_sorted = ref_channs_i # SIMPLER IMPLEMENTATION THAT LEAVES CHANNELS IN THE ORDER THEY ARE SPECIFIED...
        #... IN SETTINGS.JSON
    new_channs_sorted = [new_channs[i] for i in channs_i_sorted] # sorts the names of the channels
    reref_data_sorted = [reref_data[i] for i in channs_i_sorted] # sorts the data itself
    channs_type_sorted = [channs_type[i] for i in channs_i_sorted] # sorts the type of the data
    reref_type_sorted = [reref_type[i] for i in channs_i_sorted] # sorts the type of rereferencing
    new_coords_sorted = [new_coords[i] for i in channs_i_sorted] # sorts the channel coordinates

    # Makes a new Raw object based on the rereferenced data
    raw_info = mne.create_info(ch_names=new_channs_sorted, sfreq=info['sfreq'], ch_types=channs_type_sorted)
    raw = mne.io.RawArray(data=reref_data_sorted, info=raw_info)

    # Additional data information that cannot be included in raw.info
    extra_info = {}
    extra_info['ch_coords'] = new_coords_sorted
    extra_info['reref_type'] = reref_type_sorted


    return raw, extra_info



def epoch_data(raw, extra_info, epoch_len, include_shuffled=True):
    """ Epochs the data.

    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data to be epoched.

    extra_info : dict
    -   A dictionary containing additional information about the data that cannot be included in raw.info

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
        for ch_name in epoched_shuffled.ch_names:
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
                extra_info['reref_type'].append(extra_info['reref_type'][epoched.info.ch_names.index(ch_name)])
                extra_info['ch_coords'].append(extra_info['ch_coords'][epoched.info.ch_names.index(ch_name)])
        
        epoched.add_channels(shuffled_epochs)
        extra_info['data_type'].extend(list(np.repeat('shuffled', n_shuffles*len(epoched_shuffled.ch_names))))

    
    return epoched, extra_info



def process(raw, epoch_len, annotations=None, channels=None, coords=[], rereferencing=None, resample=None,
            highpass=None, lowpass=None, notch=None, include_shuffled=True, verbose=True):
    """ Preprocessing of the raw data in preparation for coherence analysis.
    
    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data to be processed.

    epoch_len : int | float
    -   The duration (in seconds) of segments to epoch the data into.

    annotations : MNE Annotations object | None (default)
    -   The annotations to eliminate BAD segments from the data. If None (default), no segments are excluded.

    channels : list of str | None (default)
    -   The names of the channels to pick from the data. If None (default), all channels in raw are chosen.

    coords: list of lists
    -   The coordinates of the picked channels. If an empty list (default), the coordinates from raw are used.

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
    raw_data = raw.get_data(reject_by_annotation='omit').copy() # the data itself

    # Gets channel coordinates from the Raw object
    ch_coords = raw._get_channel_positions().copy().tolist() # coordinates of the channels
    if coords == []: # if no coordinates are specified, use those from raw
        if np.isnan(np.mean(ch_coords)): # makes sure coordinates are present and not just NaN
            raise ValueError("There are missing coordinate values for the channels.")
        for ch_i, ch_coord in enumerate(ch_coords): # need to alter the units for correct plotting if taken from raw
            ch_coords[ch_i] = [ch_coord[coord_i]*1000 for coord_i in range(len(ch_coord))]
    else: # if coordinates are specified, use those instead of the ones from raw
        if np.shape(coords) != np.shape(ch_coords): # makes sure the supplied coordinates are in the correct format
            raise ValueError(f"Coordinates for the channels have been provided, but the dimensions do not match those in the original data.\nShould have shape {np.shape(ch_coords)}, but instead have shape {np.shape(coords)}.")
        ch_coords = coords


    ## Rereferencing
    if rereferencing != None:
        if verbose:
            print("Rereferencing the data")
        raw, extra_info = reref_data(raw_data, raw.info, rereferencing, ch_coords)
    else:
        extra_info = {}
        extra_info['reref_type'] = list(np.repeat('none', len(raw.info.ch_names)))
        extra_info['ch_coords'] = ch_coords
    extra_info['data_type'] = list(np.repeat('real', len(raw.info.ch_names))) # sets data as real (i.e. not shuffled;...
    #... useful in case of later comparison with shuffled data)


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
        epoched, extra_info = epoch_data(raw, extra_info, epoch_len, include_shuffled)


    return epoched, extra_info

