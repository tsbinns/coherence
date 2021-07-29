import mne
import numpy as np


def get_psd(epoched):
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
                'psd':    [],
                'freqs':   []}

    psds, freqs = mne.time_frequency.psd_welch(epoched)
    psds = np.log10(psds) * 10.
    psds = psds.mean(0)
    psd_data['psd'] = psds
    psd_data['freqs'] = np.tile(freqs, [len(used_channels),1])

    return psd_data


def get_coherence(epoched):
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
    indices = mne.connectivity.seed_target_indices(cortical, deep)

    coh_data = {'ch_name_cortical': [names[i] for i in indices[0]],
                'ch_name_deep':     [names[i] for i in indices[1]],
                'coh':              [],
                'freqs':            []}

    cohs, freqs, _, _, _ = mne.connectivity.spectral_connectivity(epoched, indices=indices, sfreq=epoched.info['sfreq'],
                                                                  mode='cwt_morlet', cwt_freqs=np.arange(0,100),
                                                                  cwt_n_cycles=len(epoched.times)/epoched.info['sfreq'])
    
    coh_data['coh'] = cohs.mean(-1)
    coh_data['freqs'] = np.tile(freqs, [len(indices[0]),1])

    return coh_data