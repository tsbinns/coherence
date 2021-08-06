import sys 
import os
from pathlib import Path
from mne_bids import read_raw_bids, BIDSPath
from mne import read_annotations
import numpy as np
import json
import pandas as pd
import matplotlib; matplotlib.use('TKAgg')


## Gets path info =====
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(MAIN_PATH, 'coherence'))
BIDS_PATH = os.path.join(MAIN_PATH, 'data')

import plotting


## Setup =====
# Loads settings for analysis
sub = '004'
task = 'Rest'
med = 'MedOff'
stim = 'StimOff'
with open(os.path.join(BIDS_PATH, 'settings.json')) as json_file:
    settings = json.load(json_file)
    run = settings['COHERENCE']['subjects'][sub]['analyses'][task][med][stim]

# Loads data
loadpath = os.path.join(BIDS_PATH, 'derivatives', 'COHERENCE', f'{task}-{sub}-{med}-{stim}-psd.pkl')
psd = pd.read_pickle(loadpath)

loadpath = os.path.join(BIDS_PATH, 'derivatives', 'COHERENCE', f'{task}-{sub}-{med}-{stim}-coh.pkl')
coh = pd.read_pickle(loadpath)


## Plotting =====
plotting.psd(psd, freq_limit=50)
plotting.coherence(coh, freq_limit=50, methods=run['coherence_methods'])