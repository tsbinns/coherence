import numpy as np
import pandas as pd
from copy import deepcopy



def check_identical(data, keys, exclude=None):

    # should checks that data[keys] have same contents (and by extension, shape)
    # should also check that all (except for excluded keys) have same shape (but not necessarily contents)

    return 'jeff'



def freq_band_info():

    bands = {
        'theta':     [4, 8],
        'alpha':     [8, 12],
        'low beta':  [13, 20],
        'high beta': [20, 35],
        #'all beta':  [13, 35],
        'gamma':     [60, 90]
    }

    return bands



def freq_band_indices(freqs, bands, include_outside=False):

    # If the requested bands did not align with the frequencies (e.g. a band of 3-5 was requested, but the frequencies
    # are [0, 2, 4, 6, 8]), include_shuffled=False would not include the indices of frequency values outside the requested band
    # (i.e. this function would return [2], corresponding to 4 Hz), whereas include_shuffled=True would include these extra
    # indices (i.e would return [1, 2, 3], corresponding to 2, 4, and 6 Hz)

    keys = bands.keys()
    idxs = deepcopy(bands)

    for key in keys:
        band_freqs = bands[key]

        if len(band_freqs) == 2:
            if band_freqs[0] < band_freqs[1]:
                if min(freqs) > band_freqs[1] or max(freqs) < band_freqs[0]: # if the frequency to index is outside the range (too low or too high)
                    idxs[key] = [np.nan, np.nan]
                    print('WARNING: The requested frequency band [%d, %d] is not contained in the data.' %(band_freqs[0], band_freqs[1]))
                else:
                    if min(freqs) > band_freqs[0]: # if the minimum frequency available is higher than lower band range, change it
                        band_freqs[0] = min(freqs)
                    if max(freqs) < band_freqs[1]: # if the maximum frequency available is lower than the higher band range, change it
                        band_freqs[1] = max(freqs)
                    if include_outside == False: # only gets frequencies within the band
                        # finds value closest to, but equal to/higher than the requested frequency
                        idxs[key][0] = int(np.where(freqs == freqs[freqs >= band_freqs[0]].min())[0])
                        # finds value closest to, but equal to/lower than the requested frequency
                        idxs[key][1] = int(np.where(freqs == freqs[freqs <= band_freqs[1]].max())[0])
                    else: # gets frequencies outside the band
                        # finds value closest to, but equal to/lower than the requested frequency
                        idxs[key][0] = int(np.where(freqs == freqs[freqs <= band_freqs[0]].max())[0])
                        # finds value closest to, but equal to/higher than the requested frequency
                        idxs[key][1] = int(np.where(freqs == freqs[freqs >= band_freqs[1]].min())[0])

            else:
                raise ValueError('The first frequency value must be lower than the second value.')
                
        else:
            raise ValueError('There should be two entries for the frequency band, not %d.' %(len(band_freqs)))

    return idxs


def find_unique_shuffled(fullnames):
    
    shuffled_idxs = []
    for i, name in enumerate(fullnames):
        if name[:8] == 'SHUFFLED':
            shuffled_idxs.append(i)

    realnames = {'name': [], 'idx': []}
    for idx in shuffled_idxs:
        realnames['name'].append(fullnames[idx][fullnames[idx].index('_')+1:])
    realnames['name'] = np.unique(realnames['name']).tolist()

    for i, name in enumerate(realnames['name']): # for each unique channel name
        realnames['idx'].append([])
        for idx in shuffled_idxs: # for each shuffled channel
            if name in fullnames[idx]: # if the channel contains the unique channel name
                realnames['idx'][i].append(idx) # put the channel indexes of the same name together

    return realnames



def average_shuffled(data, keys_to_avg, channel_name_key='ch_name'):

    """ Old method that doesn't work with coherence values
    # Find names and indexes of shuffled channels belonging to the same real channel
    shuffled = find_unique_shuffled(data[channel_name_key])

    # Averages the shuffled data across these unique channels
    avg_over = shuffled['idx'] # entries to average over
    discard = [] # entries to remove after averaging
    for over in avg_over:
        if len(over) > 1: # if there is more than one piece of data
            for key in keys_to_avg: # average the data
                data[key][over[0]] = data[key][over].mean(0)
                # SHOULD ADD CHECK TO MAKE SURE DATA IS 1D
            discard.extend(over[1:]) # adds entries to remove later
        # renames the remaining channel
        name = data[channel_name_key][over[0]]
        data[channel_name_key][over[0]] = name[:name.index('-')] + name[name.index('_'):]

    for key in data.keys():
        data[key] = np.delete(data[key], discard, 0)

    return data
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



def coherence_by_band(data, methods):

    bands = freq_band_info()

    band_data = {}
    band_data['bands'] = []
    band_data['avg'] = [] # average coherence in each frequency band
    band_data['max'] = [] # maximum coherence in each frequency band
    band_data['fmax'] = [] # frequency of maximum coherence in each frequency band
    for method_i, method in enumerate(methods):
        band_data['avg'].append([])
        band_data['max'].append([])
        band_data['fmax'].append([])

        for i in range(len(data[method])):
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