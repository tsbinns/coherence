from mne.io.brainvision.brainvision import _parse_impedance_ranges
import mne.time_frequency as tf
import mne.connectivity as con
import numpy as np
import pandas as pd
import helpers



def normalise(psds, line_noise=50, window=5):
    """ Normalises PSD data to the % total power.

    PARAMETERS
    ----------
    psds : dict
        A dictionary containing PSD data with keys 'psd' and 'freqs'. 'psd' is an array of shape (n_channels,
        n_frequencies) containing power information. 'freqs' is an array of shape (n_channels, n_frequencies) containing
        the corresponding frequencies (in Hz) for the data in 'psd'.
    line_noise : int | float
        The frequency (in Hz) of line noise in the recordings. 50 Hz by default.
    window : int | float
        The frequency (in Hz) around specific frequencies for which the power is omitted from the normalisation. 5 Hz by
        default. The specific frequencies are 0 Hz (for low-frequency noise), and the line noise (plus its harmonics,
        e.g. 50, 100, and 150 Hz). The window is applied to both sides of these frequencies (e.g. a window of 5 Hz would
        result in the frequencies 45-55 Hz to be ommited).

    RETURNS
    ----------
    psds : dictionary
        The normalised PSD data.

    """

    for psd_i in range(len(psds['psd'])): # for each channel of data

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
    """ Calculates the PSDs of the epoched data using wavelets.
    
    PARAMETERS
    ----------
    epoched : MNE Epoch object
        The epoched data to analyse.
    l_freq : int | float
        The lowest frequency to calculate the power of. 0 Hz by default.
    h_freq : int | float
        The highest frequency to calculate the power of. 100 Hz by default.
    norm : bool, default True
        Whether or not to normalise the power values to the % of the maximum total power. True by default.
    line_noise : int | float
        The frequency (in Hz) of the line noise in the recording. 50 Hz by default.

    RETURNS
    ----------
    psd_data : pandas DataFrame
        A DataFrame containing the channel names, their types (e.g. cortical, deep), and the power of the data in these
        channels.
    psd_keys : dict
        A dictionary containing the keys of the psd_data DataFrame with keys 'x' and 'y'. 'x' contains keys for
        independent/control variables, and 'y' keys for dependent variables.

    """

    # Gets channels to analyse
    types = epoched.get_channel_types()
    names = epoched.ch_names
    cortical = [] # index of ECoG channels
    deep = [] # index of LFP channels
    for i, type in enumerate(types):
        if type == 'ecog':
            cortical.append(i)
            types[i] = 'cortical'
        elif type == 'dbs':
            deep.append(i)
            types[i] = 'deep'
    used_channels = [*cortical, *deep]
    ch_names = [names[i] for i in used_channels]
    ch_types = [types[i] for i in used_channels]

    # Gets PSDs
    freqs = np.arange(l_freq, h_freq+1) # frequencies to analyse
    psds = tf.tfr_morlet(epoched, freqs, n_cycles=9, return_itc=False, average=False)
    psds = psds.data.mean(-1) # averages over time points in each epoch
    psds = np.transpose(psds, (1,2,0)) # changes dimensions to channels x freqs x epochs
    psds = psds.mean(-1) # averages over epochs
    freqs = np.tile(freqs, [len(used_channels),1])

    # Collects data
    psd_data = list(zip(ch_names, ch_types, freqs, psds))
    psd_data = pd.DataFrame(data=psd_data, columns=['ch_name', 'ch_type', 'freqs', 'psd'])

    # Average shuffled LFP values
    psd_data = helpers.average_shuffled(psd_data, ['psd', 'freqs'])

    # Normalise PSDs
    if norm is True:
        psd_data = normalise(psd_data, line_noise)

    # Gets keys in psd
    psd_keys = {
        'x': ['ch_name', 'ch_type', 'freqs'], # independent & control variables
        'y': ['psd'] # dependent variables
        }


    return psd_data, psd_keys



def get_coherence(epoched, cwt_freqs, methods=['coh', 'imcoh']):
    """ Calculates the coherence of the epoched data using wavelets.
    
    PARAMETERS
    ----------
    epoched : MNE Epoch object
        The epoched data to analyse.
    cwt_freqs : list of int | list of float
        Frequencies (in Hz) to be used in the wavelet-based coherence analysis.
    methods : list of str
        Methods for calculating the coherence of the data. By default, 'coh' (standard coherence) and 'imcoh'
        (imaginary coherence).

    RETURNS
    ----------
    coh_data : pandas DataFrame
        A DataFrame containing the names of the ECoG and LFP channel pairs used to calculate coherence, the single-
        frequency-wise coherence data, and the frequency band-wise coherence data.
    coh_keys : dict
        A dictionary containing the keys of the coh_data DataFrame with keys 'x' and 'y'. 'x' contains keys for
        independent/control variables, and 'y' keys for dependent variables.

    """

    # Gets the channels to analyses
    types = epoched.get_channel_types()
    names = epoched.ch_names
    cortical = [] # index of ECoG channels
    deep = [] # index of LFP channels
    for i, type in enumerate(types):
        if type == 'ecog':
            cortical.append(i)
            types[i] = 'cortical'
        elif type == 'dbs':
            deep.append(i)
            types[i] = 'deep'
    indices = con.seed_target_indices(cortical, deep)
    ch_names_cortical = [names[i] for i in indices[0]]
    ch_names_deep = [names[i] for i in indices[1]]

    # Gets frequency-wise coherence
    cohs = []
    for method in methods:
        coh, freqs, _, _, _ = con.spectral_connectivity(epoched, method=method, indices=indices,
                                                        sfreq=epoched.info['sfreq'], mode='cwt_morlet',
                                                        cwt_freqs=cwt_freqs, cwt_n_cycles=9)
        if method == 'imcoh':
            coh = np.abs(coh)
        cohs.append(np.asarray(coh).mean(-1))
    freqs = np.tile(freqs, [len(indices[0]),1])

    # Collects data
    coh_data = list(zip(ch_names_cortical, ch_names_deep, freqs, *cohs[:]))
    coh_data = pd.DataFrame(data=coh_data, columns=['ch_name_cortical', 'ch_name_deep', 'freqs', *methods])

    # Gets band-wise coherence
    coh_data = helpers.coherence_by_band(coh_data, methods, band_names=['theta','alpha','low beta','high beta','gamma'])

    # Average shuffled LFP values
    fbands_keynames = ['fbands_avg_', 'fbands_max_', 'fbands_fmax_']
    fbands_keys = []
    for method in methods:
        for keyname in fbands_keynames:
            fbands_keys.append(keyname+method)
    coh_data = helpers.average_shuffled(coh_data, [*methods, *fbands_keys], 'ch_name_deep')
    
    # Gets keys in psd
    coh_keys = {
        'x': ['ch_name_cortical', 'ch_name_deep', 'freqs', 'fbands'], # independent & control variables
        'y': [*methods, *fbands_keys] # dependent variables
        }


    return coh_data, coh_keys

