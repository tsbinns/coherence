import numpy as np
import pandas as pd
from copy import deepcopy



def check_identical(data, keys, exclude=None):

    # should checks that data[keys] have same contents (and by extension, shape)
    # should also check that all (except for excluded keys) have same shape (but not necessarily contents)

    return 'jeff'



def freq_band_info(names=None):
    """ Dictionary of frequency bands and their corresponding frequencies (in Hz).

    PARAMETERS
    ----------
    names : list of str | None (default)
        The names of bands to be returned. If None (default), all bands are included in the output.

    RETURNS
    ----------
    bands : dict
        A dictionary consisting of key-value pairs for frequency band names and their corresponding frequencies (in Hz).

    """

    bands = {
        'theta':     [4, 8],
        'alpha':     [8, 12],
        'low beta':  [13, 20],
        'high beta': [20, 35],
        'all beta':  [13, 35],
        'gamma':     [60, 90]
    }

    # Returns requested bands, if appropriate
    keep_keys = [] # finds which bands should be kept
    if names != None:
        for name in names:
            if name in bands.keys():
                keep_keys.append(name)
            else:
                raise ValueError(f'The requested band {name} is not a recognised frequency band.')
    discard = [] # removes the unrequested bands
    for band in list(bands.keys()):
        if band not in keep_keys:
            bands.pop(band)


    return bands



def freq_band_indices(freqs, bands, include_outside=False):
    """ Finds the indices of frequency bands in data.

    PARAMETERS
    ----------
    freqs : list of int | list of float
        The frequencies that you want to index.
    bands : dict with values of shape (n_bands x 2)
        A dictionary containing the frequency bands to be indexed. Each value should have a length of 2, i.e. a lower
        and a higher frequency to index.
    include_outside : bool, default False
        Whether or not to index frequencies outside the requested bands in the events that the bands do not align with
        the available frequencies. E.g. Say a band of 3-5 was requested, but the frequencies are [0, 2, 4, 6, 8]),
        include_shuffled=False would not include the indices of frequency values outside the requested band (i.e. this
        function would return [2], corresponding to 4 Hz), whereas include_shuffled=True would include these extra
        indices (i.e would return [1, 2, 3], corresponding to 2, 4, and 6 Hz).

    RETURNS
    ----------
    idc : dict of shape (n_bands x 2)
        A dictionary containing values of the indices of the lower and upper frequency band limits.

    """

    keys = bands.keys()
    idc = deepcopy(bands)

    for key in keys:
        band_freqs = bands[key]
        if len(band_freqs) == 2:
            if band_freqs[0] < band_freqs[1]:
                if min(freqs) > band_freqs[1] or max(freqs) < band_freqs[0]: # if the frequency to index is outside...
                #... the range (too low or too high)
                    idc[key] = [np.nan, np.nan]
                    print(f'WARNING: The requested frequency band [{band_freqs[0]}, {band_freqs[1]}] is not contained in the data, setting indeces to NaN.')
                else:
                    if min(freqs) > band_freqs[0]: # if the minimum frequency available is higher than lower band...
                    #... range, change it
                        band_freqs[0] = min(freqs)

                    if max(freqs) < band_freqs[1]: # if the maximum frequency available is lower than the higher band...
                    #... range, change it
                        band_freqs[1] = max(freqs)

                    if include_outside == False: # only gets frequencies within the band
                        # finds value closest to, but equal to/higher than the requested frequency
                        idc[key][0] = int(np.where(freqs == freqs[freqs >= band_freqs[0]].min())[0])
                        # finds value closest to, but equal to/lower than the requested frequency
                        idc[key][1] = int(np.where(freqs == freqs[freqs <= band_freqs[1]].max())[0])
                    else: # gets frequencies outside the band
                        # finds value closest to, but equal to/lower than the requested frequency
                        idc[key][0] = int(np.where(freqs == freqs[freqs <= band_freqs[0]].max())[0])
                        # finds value closest to, but equal to/higher than the requested frequency
                        idc[key][1] = int(np.where(freqs == freqs[freqs >= band_freqs[1]].min())[0])
            else:
                raise ValueError('The first frequency value must be lower than the second value.')  
        else:
            raise ValueError(f'There should be two entries for the frequency band, not {band_freqs}.')


    return idc



def find_unique_shuffled(fullnames):
    """ Finds the names and indices of shuffled channels, discarding duplicates.

    PARAMETERS
    ----------
    fullnames : list of str
        The names of channels to be filtered.

    RETURNS
    ----------
    realnames : dict
        A dictionary with keys 'name' and 'idx'. 'name' contains the unique channel names (with the shuffled affix
        removed). 'idx' contains the indices of each unique shuffled channel in the variable fullnames.

    """
    
    # Finds the shuffled channels
    shuffled_idc = []
    for i, name in enumerate(fullnames):
        if name[:8] == 'SHUFFLED':
            shuffled_idc.append(i)

    # Removes the shuffled affix from the channel names, then removes duplicate channel names
    realnames = {'name': [], 'idx': []}
    for idx in shuffled_idc:
        realnames['name'].append(fullnames[idx][fullnames[idx].index('_')+1:])
    realnames['name'] = np.unique(realnames['name']).tolist()

    # Finds where these unique channels are present
    for i, name in enumerate(realnames['name']): # for each unique channel name
        realnames['idx'].append([])
        for idx in shuffled_idc: # for each shuffled channel
            if name in fullnames[idx]: # if the channel contains the unique channel name
                realnames['idx'][i].append(idx) # put the channel indexes of the same name together


    return realnames



def average_shuffled(data, keys_to_avg, channel_name_key='ch_name'):
    """ Averages data over adjacent shuffled channels of the same type.

    PARAMETERS
    ----------
    data : pandas DataFrame
        The data to average over. Must contain a column that can be used to identify the shuffled channels.
    keys_to_avg : list of str
        The keys of the DataFrame columns to be averaged.
    channel_name_key : str
        The name of the column used to identify the shuffled channels. 'ch_name' by default.

    RETURNS
    ----------
    data : pandas DataFrame
        The data with entries from adjacent shuffled channels of the same type averaged over.

    """

    # Find the unique shuffled channels based on their positions
    i = 0
    avg_over = []
    for idx, name in enumerate(data[channel_name_key]): # for each channel
        if name[:9] != 'SHUFFLED-': # if it's not a shuffled channel, move on
            avg_over.append([])
            i += 1
        else:
            if name[:10] == 'SHUFFLED-0': # if it's the start of a shuffled channel run, add it to the index
                avg_over.append([])
                avg_over[i] = [idx]
            else: # and add any subsequent channels
                avg_over[i].extend([idx])

    # Excludes the empty entries from avg_over
    empty = []
    for i in range(len(avg_over)):
        if not avg_over[i]:
            empty.append(i)
    avg_over = np.delete(avg_over, empty, 0).tolist()

    # Averages the shuffled data across the unique channels
    discard = [] # entries to remove after averaging
    for over in avg_over:
        if len(over) > 1: # if there is more than one piece of data
            for key in keys_to_avg: # average the data
                data[key][over[0]] = np.mean(data[key][over].tolist(), axis=0)
                # SHOULD ADD CHECK TO MAKE SURE DATA IS 1D
            discard.extend(over[1:]) # adds entries to remove later
        # renames the remaining channel
        name = data[channel_name_key][over[0]]
        data[channel_name_key][over[0]] = name[:name.index('-')] + name[name.index('_'):]
    data.drop(discard, inplace=True) # deletes the redundant entries
    data.reset_index(inplace=True)


    return data



def coherence_by_band(data, methods, band_names=None):
    """ Calculates band-wise coherence values based on the frequency-wise data.

    PARAMETERS
    ----------
    data : pandas DataFrame
        The frequency-wise coherence data.
    methods : list of str
        The methods used to calculate coherence.
    band_names : list of str | None (default)
        The frequency bands to calculate band-wise coherence values for. If None (default), all frequency bands provided
        by the helpers.freq_band_info function are used.
    
    RETURNS
    ----------
    data : pandas DataFrame
        The original frequency-wise coherence data, plus band-wise coherence values. These values include: the average
        coherence in each band (avg); the maximum coherence in each band (max); and the frequency at which this maximum
        coherence occurs (fmax).

    """

    # Gets the band-wise data
    bands = freq_band_info(band_names) # the frequency bands to analyse

    band_data = {}
    band_data['bands'] = []
    band_data['avg'] = [] # average coherence in each frequency band
    band_data['max'] = [] # maximum coherence in each frequency band
    band_data['fmax'] = [] # frequency of maximum coherence in each frequency band
    for method_i, method in enumerate(methods): # for each coherence calculation method
        band_data['avg'].append([])
        band_data['max'].append([])
        band_data['fmax'].append([])

        for i in range(len(data[method])): # finds the band-wise coherence for each channel
            band_is = freq_band_indices(data['freqs'][i], bands)
            if method_i == 0:
                band_data['bands'].append(list(bands.keys()))
            band_data['avg'][method_i].append([])
            band_data['max'][method_i].append([])
            band_data['fmax'][method_i].append([])
            for key in bands.keys():
                band_data['avg'][method_i][i].append(data[method][i][band_is[key][0]:band_is[key][1]+1].mean())
                band_data['max'][method_i][i].append(data[method][i][band_is[key][0]:band_is[key][1]+1].max())
                band_data['fmax'][method_i][i].append(data['freqs'][i][int(np.where(data[method][i] == 
                                                      band_data['max'][method_i][i][-1])[0])])
    
    # Collects band-wise data
    band_data = list(zip(band_data['bands'], *band_data['avg'][:], *band_data['max'][:], *band_data['fmax'][:]))
    fbands_keynames = ['fbands_avg_', 'fbands_max_', 'fbands_fmax_']
    fbands_keys = []
    for keyname in fbands_keynames:
        for method in methods:
            fbands_keys.append(keyname+method)
    band_data = pd.DataFrame(data=band_data, columns=['fbands', *fbands_keys])

    # Collects frequency- and band-wise data
    data = pd.concat([data, band_data], axis=1)


    return data