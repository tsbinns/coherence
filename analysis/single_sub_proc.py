from copy import deepcopy
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


# Gets path info
cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))

main_path = 'C:\\Users\\tomth\\OneDrive\\Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata'
project_path = os.path.join(main_path, 'projects', 'coherence')



import preprocessing
import processing
from helpers import combine_data


## Setup =====
# Loads settings for analysis
sub = '004'
task = 'Rest'
med = 'MedOff'
stim = 'StimOff'
with open(os.path.join(project_path, 'settings.json')) as json_file:
    settings = json.load(json_file)
    run = settings['subjects'][sub]['analyses'][task][med][stim]

notch = np.arange(run['line_noise'], run['lowpass']+1, run['line_noise']) # Hz
wavelet_freqs = np.arange(run['highpass'], run['psd_highfreq']+1)


## Analysis =====
data_paths = run['data_paths']
annot_paths = []
for annot_path in run['annotation_paths']:
    annot_paths.append(os.path.join(project_path, annot_path))
all_psds = []
all_cohs = []
for i, run_i in enumerate(data_paths):
    # Load data
    bids_path = BIDSPath(data_paths[run_i]['subject'], data_paths[run_i]['session'], data_paths[run_i]['task'],
                         data_paths[run_i]['acquisition'], data_paths[run_i]['run'], root=main_path)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    annots = read_annotations(annot_paths[i])

    # Process data
    processed = preprocessing.process(raw, run['epoch_length'], annotations=annots, channels=run['channels'],
                                      resample=run['resample'], highpass=run['highpass'], lowpass=run['lowpass'],
                                      notch=notch)
    psd, psd_keys = processing.get_psd(processed, run['psd_lowfreq'], run['psd_highfreq'],
                    line_noise=run['line_noise'])
    coh, coh_keys = processing.get_coherence(processed, wavelet_freqs, run['coherence_methods'])

    # Collects data
    all_psds.append(psd)
    all_cohs.append(coh)

# Combines data
data_info = {
    'run': list(data_paths.keys()),
    'subject': [sub],
    'task': [task],
    'stim': [stim],
    'med': [med]
}
psds = deepcopy(all_psds)
cohs = deepcopy(all_cohs)
for i in range(len(data_info)):
    info = {list(data_info.keys())[i]: list(data_info.values())[i]}
    if len(list(info.values())[0]) == 1:
        psds = [psds]
        cohs = [cohs]
    psds = combine_data(psds, info)
    cohs = combine_data(cohs, info)

# Saves data
savepath = os.path.join(project_path, 'derivatives', f'{task}-{sub}-{med}-{stim}-psd.pkl')
psds.to_pickle(savepath)

savepath = os.path.join(project_path, 'derivatives', f'{task}-{sub}-{med}-{stim}-coh.pkl')
cohs.to_pickle(savepath)

