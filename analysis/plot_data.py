import sys 
import os
from pathlib import Path
from mne_bids import read_raw_bids, BIDSPath
from mne import read_annotations
import numpy as np
import json
import pandas as pd
import matplotlib; matplotlib.use('TKAgg')


### Gets path info
cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))

main_path = 'C:\\Users\\tomth\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata'
project_path = 'C:\\Users\\tomth\\OneDrive\\My Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\projects\\coherence'

import plotting
from helpers import average_dataset


### Setup & Processing
# Loads data
datasets = ['Rest-003-MedOff-StimOff', 'Rest-003-MedOn-StimOff']

psds = []
cohs = []
for data in datasets:
    psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-psd.pkl')))
    cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-coh.pkl')))

    # Averages data over runs
    psds[-1] = average_dataset(data=psds[-1], avg_over='run', ch_keys=['ch_name'],
                               x_keys=['med','stim','task','subject','ch_type','freqs'], y_keys=['psd'])

    cohs[-1] = average_dataset(data=cohs[-1], avg_over='run', ch_keys=['ch_name_cortical', 'ch_name_deep'],
                               x_keys=['med','stim','task','subject','freqs','fbands'],
                               y_keys=['coh', 'imcoh', 'coh_fbands_avg', 'imcoh_fbands_avg', 'coh_fbands_max',
                                       'imcoh_fbands_max', 'coh_fbands_fmax', 'imcoh_fbands_fmax'])

psd = pd.concat(psds[:], ignore_index=True)
coh = pd.concat(cohs[:], ignore_index=True)


### Plotting
plotting.psd(psd, plot_shuffled=False, plot_std=False, freq_limit=50, power_limit=15)
plotting.coherence_fwise(coh, plot_shuffled=True, plot_std=False, freq_limit=50, methods=['coh', 'imcoh'])
plotting.coherence_bandwise(coh, plot_shuffled=False, plot_std=False, methods=['coh', 'imcoh'])