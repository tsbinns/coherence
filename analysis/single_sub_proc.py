import sys 
import os
from pathlib import Path
from mne_bids import read_raw_bids, BIDSPath
from mne import read_annotations
import numpy as np
import json
import pandas as pd
from copy import deepcopy
import matplotlib; matplotlib.use('TKAgg')


## Gets path info =====
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(MAIN_PATH, 'coherence'))
BIDS_PATH = os.path.join(MAIN_PATH, 'data')

import preprocessing
import processing
from helpers import check_identical


## Setup =====
# Loads settings for analysis
sub = '004'
task = 'Rest'
med = 'MedOff'
stim = 'StimOff'
with open(os.path.join(BIDS_PATH, 'settings.json')) as json_file:
    settings = json.load(json_file)
    run = settings['COHERENCE']['subjects'][sub]['analyses'][task][med][stim]

notch = np.arange(run['line_noise'], run['lowpass']+1, run['line_noise']) # Hz
wavelet_freqs = np.arange(run['highpass'], run['psd_highfreq']+1)


## Analysis =====
data_paths = run['data_paths']
annot_paths = run['annotation_paths']
psds = []
cohs = []
for i in data_paths:
    # Load data
    bids_path = BIDSPath(data_paths[i]['subject'], data_paths[i]['session'], data_paths[i]['task'],
                         data_paths[i]['acquisition'], data_paths[i]['run'], root=BIDS_PATH)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    annots = read_annotations(annot_paths[int(i)])

    # Process data
    processed = preprocessing.process(raw, run['epoch_length'], annotations=annots, channels=run['channels'],
                                      resample=run['resample'], highpass=run['highpass'], lowpass=run['lowpass'],
                                      notch=notch)
    psd, psd_keys = processing.get_psd(processed, run['psd_lowfreq'], run['psd_highfreq'],
                    line_noise=run['line_noise'])
    psds.append(psd)
    coh, coh_keys = processing.get_coherence(processed, wavelet_freqs, run['coherence_methods'])
    cohs.append(coh)

# Checks data is identical between runs
check_identical(psds, psd_keys['x'])
check_identical(cohs, coh_keys['x'])

# Averages data
avg_psd = psds[0].copy()
for data_i in range(len(avg_psd.ch_name)):
    for key in psd_keys['y']:
        data = []
        for psd_i in range(len(psds)):
            data.append(psds[psd_i][key][data_i])
        avg_psd[key][data_i] = np.mean(data, 0)
    
avg_coh = cohs[0].copy()
for data_i in range(len(avg_coh['ch_name_cortical'])):
    for key in coh_keys['y']:
        data = []
        for coh_i in range(len(cohs)):
            data.append(cohs[coh_i][key][data_i])
        avg_coh[key][data_i] = np.mean(data, 0)

# Saves data
savepath = os.path.join(BIDS_PATH, 'derivatives', 'COHERENCE', f'{task}-{sub}-{med}-{stim}-psd.pkl')
avg_psd.to_pickle(savepath)

savepath = os.path.join(BIDS_PATH, 'derivatives', 'COHERENCE', f'{task}-{sub}-{med}-{stim}-coh.pkl')
avg_coh.to_pickle(savepath)

