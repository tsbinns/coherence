from mne.io.brainvision.brainvision import _parse_impedance_ranges
import mne.time_frequency as tf
import mne.connectivity as con
import numpy as np
import helpers



def normalise(psds, line_noise=50, window=5):
    """ Normalises PSD data to the % total power.

    PARAMETERS
    ----------
    psds : dictionary
        A dictionary containing PSD data with keys 'psd' and 'freqs'. 'psd' is an array of shape (n_channels,
        n_frequencies) containing power information. 'freqs' is an array of shape (n_channels, n_frequencies) containing
        the corresponding frequencies (in Hz) for the data in 'psd'.
    line_noise : int | float
        The frequency (in Hz) of line noise in the recordings. 50 Hz by default.
    window : int | float
        The frequency (in Hz) around specific frequencies for which the power is omitted from the normalisation. The
        specific frequencies are 0 Hz (for low-frequency noise), and the line noise (plus its harmonics, e.g. 50, 100,
        and 150 Hz). The window is applied to both sides of these frequencies (e.g. a window of 5 Hz would result in the
        frequencies 45-55 Hz to be ommited).

    RETURNS
    ----------
    psds : dictionary
        The normalised PSD data.
    """

    for psd_i, psd in enumerate(psds['psd']): # for each channel of data

        exclude_freqs = np.arange(0, psds['freqs'][psd_i][-1]+1, line_noise) # low and line noise frequencies to exclude
        exclude_window = {} # windows around these line noise frequencies
        for freq in exclude_freqs:
            exclude_window[freq] = [freq-window, freq+window]
        window_idc = helpers.freq_band_indices(psds['freqs'][psd_i], exclude_window, include_outside=True) # indices... 
        # ... of these windows

        exclude_idc = [] # line noise-associated frequency indices to exclude
        for wind in window_idc.values():
            if not np.isnan(np.sum(wind)):
                exclude_idc.extend(np.arange(wind[0], wind[1]+1))
        
        keep_idc = [] # indices of frequencies to normalise data
        freq_idc = np.arange(len(psds['freqs'][psd_i])) # indices of frequencies in the data
        keep_idc = [x for x in freq_idc if x not in exclude_idc]

        # Normalises data to % total power
        psds['psd'][psd_i] = (psds['psd'][psd_i] / np.sum(psds['psd'][psd_i][keep_idc]))*100
        

    return psds



def get_psd(epoched, l_freq=0, h_freq=100, norm=True, line_noise=50):
    """ Calculates the PSDs of the epoched data.
    
    PARAMETERS
    ----------


    RETURNS
    ----------
    
    """
    types = epoched.get_channel_types()
    names = epoched.ch_names
    cortical = [] # index of ECoG channels
    deep = [] # index of LFP channels
    for i, type in enumerate(types):
        if type == 'ecog':
            cortical.append(i)
            types[i] = 'cortical'
        elif type == 'seeg':
            deep.append(i)
            types[i] = 'deep'
    used_channels = [*cortical, *deep]

    psd_data = {'ch_name': [names[i] for i in used_channels],
                'ch_type': [types[i] for i in used_channels],
                'psd':     [],
                'freqs':   []}

    freqs = np.arange(l_freq, h_freq+1)
    psds = tf.tfr_morlet(epoched, freqs, n_cycles=9, return_itc=False, average=False)

    """
    psds, freqs = tf.psd_welch(epoched)
    psds = np.log10(psds) * 10.
    psds = psds.mean(0)
    """

    psds = psds.data.mean(-1) # averages over time points in each epoch
    psds = np.transpose(psds, (1,2,0)) # changes dimensions to channels x freqs x epochs
    psd_data['psd'] = psds.mean(-1) # averages over epochs
    psd_data['freqs'] = np.tile(freqs, [len(used_channels),1])

    # Average shuffled LFP values
    psd_data = helpers.average_shuffled(psd_data, ['psd', 'freqs'])

    # Normalise PSDs
    if norm is True:
        psd_data = normalise(psd_data, line_noise)

    return psd_data


def get_coherence(epoched, cwt_freqs, methods='coh'):
    types = epoched.get_channel_types()
    names = epoched.ch_names
    cortical = [] # index of ECoG channels
    deep = [] # index of LFP channels
    for i, type in enumerate(types):
        if type == 'ecog':
            cortical.append(i)
            types[i] = 'cortical'
        elif type == 'seeg':
            deep.append(i)
            types[i] = 'deep'
    indices = con.seed_target_indices(cortical, deep)

    coh_data = {'ch_name_cortical': [names[i] for i in indices[0]],
                'ch_name_deep':     [names[i] for i in indices[1]],
                'coh':              [],
                'freqs':            []}

    cohs = []
    coh, freqs, _, _, _ = con.spectral_connectivity(epoched, method=method, indices=indices, sfreq=epoched.info['sfreq'],
                                                    mode='cwt_morlet', cwt_freqs=cwt_freqs, cwt_n_cycles=9)
    cohs = np.abs(np.asarray(coh).mean(-1))
    coh_data['coh'] = cohs
    coh_data['freqs'] = np.tile(freqs, [len(indices[0]),1])

    # Finds the peak and average coherence values in each frequency band
    coh_data = helpers.coherence_by_band(coh_data)

    # Average shuffled LFP values
    coh_data = helpers.average_shuffled(coh_data, ['coh', 'freqs', 'fbands_avg', 'fbands_max', 'fbands_fmax'], 'ch_name_deep')
    

    return coh_data