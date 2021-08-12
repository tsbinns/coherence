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

main_path = 'C:\\Users\\tomth\\OneDrive\\Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata'
project_path = os.path.join(main_path, 'projects', 'coherence')

import plotting
from helpers import average_dataset


### Setup
# Loads settings for analysis
sub = '004'
task = 'Rest'
med = 'MedOff'
stim = 'StimOff'
with open(os.path.join(project_path, 'settings.json')) as json_file:
    settings = json.load(json_file)
    run = settings['subjects'][sub]['analyses'][task][med][stim]

# Loads data
loadpath = os.path.join(project_path, 'derivatives', f'{task}-{sub}-{med}-{stim}-psd.pkl')
psd = pd.read_pickle(loadpath)

loadpath = os.path.join(project_path, 'derivatives', f'{task}-{sub}-{med}-{stim}-coh.pkl')
coh = pd.read_pickle(loadpath)


### Processing
# Averages data over runs
psd = average_dataset(data=psd, avg_over='run', ch_keys=['ch_name'],
                      x_keys=['med','stim','task','subject','ch_type','freqs'], y_keys=['psd'])

coh = average_dataset(data=coh, avg_over='run', ch_keys=['ch_name_cortical', 'ch_name_deep'],
                      x_keys=['med','stim','task','subject','freqs','fbands'],
                      y_keys=['coh', 'imcoh', 'coh_fbands_avg', 'imcoh_fbands_avg', 'coh_fbands_max',
                              'imcoh_fbands_max', 'coh_fbands_fmax', 'imcoh_fbands_fmax'])


### Plotting
plotting.psd(psd, plot_shuffled=False, plot_std=False, freq_limit=50)
plotting.coherence_fwise(coh, plot_shuffled=True, plot_std=False, freq_limit=50, methods=run['coherence_methods'])
plotting.coherence_bandwise(coh, plot_shuffled=False, plot_std=True, methods=run['coherence_methods'])