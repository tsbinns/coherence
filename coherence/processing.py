import mne.time_frequency as tf
import mne.connectivity as con
import numpy as np
import pandas as pd
import helpers
from fooof import FOOOF



def normalise(psds, line_noise=50, window=5):
    """ Normalises PSD data to the % total power.

    PARAMETERS
    ----------
    psds : dict
    -   A dictionary containing PSD data with keys 'psd' and 'freqs'. 'psd' is an array of shape (n_channels,
        n_frequencies) containing power information. 'freqs' is an array of shape (n_channels, n_frequencies) containing
        the corresponding frequencies (in Hz) for the data in 'psd'.

    line_noise : int | float
    -   The frequency (in Hz) of line noise in the recordings. 50 Hz by default.

    window : int | float
    -   The frequency (in Hz) around specific frequencies for which the power is omitted from the normalisation. 5 Hz by
        default. The specific frequencies are 0 Hz (for low-frequency noise), and the line noise (plus its harmonics,
        e.g. 50, 100, and 150 Hz). The window is applied to both sides of these frequencies (e.g. a window of 5 Hz would
        result in the frequencies 45-55 Hz to be ommited).


    RETURNS
    ----------
    psds : dictionary
    -   The normalised PSD data.
    """

    for psd_i in range(psds.shape[0]): # for each channel of data

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
        psd_keys = ['power', 'power_periodic', 'power_aperiodic']
        for psd_key in psd_keys:
            psds[psd_key][psd_i] = (psds[psd_key][psd_i] / np.sum(psds[psd_key][psd_i][keep_idc]))*100
        

    return psds



def get_psd(epoched, extra_info, l_freq=0, h_freq=100, norm=True, line_noise=50):
    """ Calculates the PSDs of the epoched data using wavelets.
    
    PARAMETERS
    ----------
    epoched : MNE Epoch object
    -   The epoched data to analyse.

    extra_info : dict
    -   A dictionary containing extra information about the data that cannot be included in epoched.info.

    l_freq : int | float
    -   The lowest frequency to calculate the power of. 0 Hz by default.

    h_freq : int | float
    -   The highest frequency to calculate the power of. 100 Hz by default.

    norm : bool, default True
    -   Whether or not to normalise the power values to the % of the maximum total power. True by default.

    line_noise : int | float
    -   The frequency (in Hz) of the line noise in the recording. 50 Hz by default.


    RETURNS
    ----------
    psd_data : pandas DataFrame
    -   A DataFrame containing the channel names, their types (e.g. cortical, deep), and the power of the data in these
        channels.
    """

    # Gets channels to analyse
    # Setup
    ch_types = epoched.get_channel_types()
    ch_names = epoched.ch_names.copy()
    ch_coords = extra_info['ch_coords'].copy()
    data_types = extra_info['data_type'].copy()
    reref_types = extra_info['reref_type'].copy()

    # Channel types
    cortical = [] # index of ECoG channels
    deep = [] # index of LFP channels
    for i, type in enumerate(ch_types):
        if type == 'ecog':
            cortical.append(i)
            ch_types[i] = 'cortical'
        elif type == 'dbs':
            deep.append(i)
            ch_types[i] = 'deep'
    used_channels = [*cortical, *deep]

    # Gets PSDs
    freqs = np.arange(l_freq, h_freq+1) # frequencies to analyse
    psds = tf.tfr_morlet(epoched, freqs, n_cycles=9, return_itc=False, average=False)
    psds = psds.data.mean(-1) # averages over time points in each epoch
    psds = np.transpose(psds, (1,2,0)) # changes dimensions to channels x freqs x epochs
    psds = psds.mean(-1) # averages over epochs

    ## FOOOFs PSDs
    psds_periodic = []
    psds_aperiodic = []
    fms_info = []
    model_params = {'peak_width_limits': [2,8]}
    for i, psd in enumerate(psds):
        # Plots PSDs and model fits so that you can check whether a 'fixed' or 'knee' fit should be used for...
        #... modelling the aperiodic component, returning the chosen model
        fm, aperiodic_mode = helpers.check_fm_fits(psd, freqs, params=model_params, title=ch_names[i], report=True)
        # Extracts the periodic and aperiodic components of the model, as well as the model information
        psds_periodic.append(fm._spectrum_flat)
        psds_aperiodic.append(fm._spectrum_peak_rm)
        fm_info = helpers.collect_fm_info(fm)
        fms_info.append(fm_info)

    freqs = np.tile(freqs, [len(used_channels), 1])

    # Renames shuffled channels to be identical for ease of further processing
    ch_names = helpers.rename_shuffled(ch_names, data_types)

    # Collects data
    psd_data = list(zip(ch_names, ch_coords, data_types, reref_types, ch_types, freqs, psds, psds_periodic,
                        psds_aperiodic, fms_info))
    psd_data = pd.DataFrame(data=psd_data, columns=['ch_name', 'ch_coords', 'data_type', 'reref_type', 'ch_type',
                                                    'freqs', 'power', 'power_periodic', 'power_aperiodic', 'fm_info'])

    # Normalise PSDs
    if norm is True:
        psd_data = normalise(psd_data, line_noise)

    # Gets band-wise power
    psd_data = helpers.data_by_band(psd_data, ['power', 'power_periodic', 'power_aperiodic'],
                                    band_names=['theta','alpha','low beta','high beta','gamma'])

    # Average shuffled LFP values
    psd_data = helpers.average_shuffled(psd_data, ['power', 'power_periodic', 'power_aperiodic', 'freqs'], ['ch_name'])


    return psd_data



def get_coherence(epoched, extra_info, cwt_freqs, methods=['coh', 'imcoh']):
    """ Calculates the coherence of the epoched data using wavelets.
    
    PARAMETERS
    ----------
    epoched : MNE Epoch object
    -   The epoched data to analyse.

    extra_info : dict
    -   A dictionary containing extra information about the data that cannot be included in epoched.info.

    cwt_freqs : list of int | list of float
    -   Frequencies (in Hz) to be used in the wavelet-based coherence analysis.

    methods : list of str
    -   Methods for calculating the coherence of the data. By default, 'coh' (standard coherence) and 'imcoh'
        (imaginary coherence).


    RETURNS
    ----------
    coh_data : pandas DataFrame
    -   A DataFrame containing the names of the ECoG and LFP channel pairs used to calculate coherence, the single-
        frequency-wise coherence data, and the frequency band-wise coherence data.
    """

    # Setup
    ch_types = epoched.get_channel_types()
    ch_names = epoched.ch_names.copy()
    ch_coords = extra_info['ch_coords'].copy()
    data_type = extra_info['data_type'].copy()
    reref_types = extra_info['reref_type'].copy()

    # Channel types
    cortical = [] # index of ECoG channels
    deep = [] # index of LFP channels
    for i, type in enumerate(ch_types):
        if type == 'ecog':
            cortical.append(i)
            ch_types[i] = 'cortical'
        elif type == 'dbs':
            deep.append(i)
            ch_types[i] = 'deep'
    indices = con.seed_target_indices(cortical, deep)

    # Types of data
    data_types_cortical = [data_type[i] for i in indices[0]]
    data_types_deep = [data_type[i] for i in indices[1]]

    # Channel names
    ch_names_cortical = [ch_names[i] for i in indices[0]]
    ch_names_deep = [ch_names[i] for i in indices[1]]
    # Renames shuffled channels to be identical for ease of further processing
    ch_names_cortical = helpers.rename_shuffled(ch_names_cortical, data_types_cortical)
    ch_names_deep = helpers.rename_shuffled(ch_names_deep, data_types_deep)

    # Types of data (real or shuffled)
    data_types = [[data_type[indices[0][i]], data_type[indices[1][i]]] for i in range(len(indices[0]))]
    for dat_i, dat_type in enumerate(data_types):
        if len(np.unique(dat_type)) <= 2:
            if 'shuffled' in dat_type:
                data_types[dat_i] = 'shuffled'
            else:
                if np.unique(dat_type) == 'real':
                    data_types[dat_i] = 'real'
                else:
                    raise ValueError(f"Only 'real' and 'shuffled' data types are accepted, but type(s) {np.unique(dat_type)} is/are provided.")
        else:
            raise ValueError(f"Only 'real' and 'shuffled' data types are accepted, but types {np.unique(dat_type)} are provided.")
    
    # Types of rereferencing
    reref_type_cortical = [reref_types[i] for i in indices[0]]
    reref_type_deep = [reref_types[i] for i in indices[1]]

    # Channel coordinates
    ch_coords_cortical = [ch_coords[i] for i in indices[0]]
    ch_coords_deep = [ch_coords[i] for i in indices[1]]

    # Gets frequency-wise coherence
    cohs = []
    coh_methods = []
    n_methods = len(methods)
    for method in methods:
        coh, freqs, _, _, _ = con.spectral_connectivity(epoched, method=method, indices=indices,
                                                        sfreq=epoched.info['sfreq'], mode='cwt_morlet',
                                                        cwt_freqs=cwt_freqs, cwt_n_cycles=9)
        if method == 'imcoh':
            coh = np.abs(coh)
        cohs.extend(np.asarray(coh).mean(-1))
        coh_methods.extend([method]*len(indices[0]))
    freqs = np.tile(freqs, (len(indices[0])*n_methods, 1))

    # Collects data
    coh_data = list(zip(np.tile(ch_names_cortical, n_methods).tolist(), np.tile(ch_names_deep, n_methods).tolist(), 
                        np.tile(ch_coords_cortical, (n_methods,1)).tolist(),
                        np.tile(ch_coords_deep, (n_methods,1)).tolist(),
                        np.tile(data_types, n_methods).tolist(), np.tile(reref_type_cortical, n_methods).tolist(),
                        np.tile(reref_type_deep, n_methods).tolist(), freqs, coh_methods, cohs))
    coh_data = pd.DataFrame(data=coh_data, columns=['ch_name_cortical', 'ch_name_deep',
                                                    'ch_coords_cortical',
                                                    'ch_coords_deep',
                                                    'data_type', 'reref_type_cortical',
                                                    'reref_type_deep', 'freqs', 'method', 'coh'])

    # Gets band-wise coherence
    coh_data = helpers.data_by_band(coh_data, ['coh'], band_names=['theta','alpha','low beta','high beta','gamma'])

    # Average shuffled LFP values
    fbands_keys = ['fbands_avg', 'fbands_max', 'fbands_fmax']
    coh_data = helpers.average_shuffled(coh_data, fbands_keys, ['ch_name_cortical', 'ch_name_deep', 'method'])


    return coh_data

