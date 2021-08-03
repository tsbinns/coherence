import mne.time_frequency as tf
import mne.connectivity as con
import numpy as np
import helpers


def get_psd(epoched, l_freq=0, h_freq=100):
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

    psds = psds.data.mean(-1)
    psd_data['psd'] = np.transpose(psds, (1,2,0))
    psd_data['freqs'] = np.tile(freqs, [len(used_channels),1])
    # Average shuffled LFP values
    psd_data = helpers.average_shuffled(psd_data, ['psd', 'freqs'])

    return psd_data


def get_coherence(epoched, cwt_freqs):
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
    coh, freqs, _, _, _ = con.spectral_connectivity(epoched, indices=indices, sfreq=epoched.info['sfreq'],
                                                    mode='cwt_morlet', cwt_freqs=cwt_freqs, cwt_n_cycles=9)
    cohs = np.asarray(coh).mean(-1)
    coh_data['coh'] = cohs
    coh_data['freqs'] = np.tile(freqs, [len(indices[0]),1])

    # Finds the peak and average coherence values in each frequency band
    coh_data = helpers.coherence_by_band(coh_data)

    # Average shuffled LFP values
    coh_data = helpers.average_shuffled(coh_data, ['coh', 'freqs', 'fbands_avg', 'fbands_max', 'fbands_fmax'], 'ch_name_deep')
    

    return coh_data