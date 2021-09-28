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

main_path = 'C:\\Users\\tomth\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata'
project_path = 'C:\\Users\\tomth\\OneDrive\\My Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\projects\\coherence'

import preprocessing
import processing
from helpers import combine_data


## Setup =====
# Loads settings for analysis
sub = '007'
task = 'Rest'
med = 'MedOn'
stim = 'StimOff'
with open(os.path.join(project_path, 'settings.json')) as json_file:
    settings = json.load(json_file)
    analysis = settings['analyses'][task]
    sub_data = settings['analyses'][task][med][stim]['subjects'][sub]

notch = np.arange(analysis['line_noise'], analysis['lowpass']+1, analysis['line_noise']) # Hz
wavelet_freqs = np.arange(analysis['highpass'], analysis['psd_highfreq']+1)


## Analysis =====
all_psds = []
all_cohs = []
for i in sub_data:
    # Load data
    bids_path = BIDSPath(sub, sub_data[i]['session'], task, sub_data[i]['acquisition'], sub_data[i]['run'],
                         root=main_path)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    annots = None
    
    try:
        annots = read_annotations(os.path.join(project_path, 'annotations', bids_path.basename+'.csv'))
    except:
        print('No annotations to read.')
    

    # Process data
    processed, extra_info = preprocessing.process(raw, analysis['epoch_length'], annotations=annots,
                                                  channels=sub_data[i]['channels'],
                                                  coords=sub_data[i]['coords'],
                                                  rereferencing=sub_data[i]['rereferencing'],
                                                  resample=analysis['resample'], highpass=analysis['highpass'],
                                                  lowpass=analysis['lowpass'], notch=notch)

    psd = processing.get_psd(processed, extra_info, analysis['psd_lowfreq'], analysis['psd_highfreq'],
                             line_noise=analysis['line_noise'])
    coh = processing.get_coherence(processed, extra_info, wavelet_freqs, analysis['coherence_methods'])

    # Collects data
    all_psds.append(psd)
    all_cohs.append(coh)

# Combines data
data_info = {
    'run': list(sub_data.keys()),
    'subject': [sub],
    'task': [task],
    'stim': [stim[4:]],
    'med': [med[3:]]
}
psds = deepcopy(all_psds)
cohs = deepcopy(all_cohs)
for i in range(len(data_info)):
    info = {list(data_info.keys())[i]: list(data_info.values())[i]}
    if len(list(info.values())[0]) == 1:
        if type(psds) is not list:
            psds = [psds]
        if type(cohs) is not list:
            cohs = [cohs]
    psds = combine_data(psds, info)
    cohs = combine_data(cohs, info)

# Saves data
savepath = os.path.join(project_path, 'derivatives', f'{task}-{sub}-{med}-{stim}')
print(f'Saving data to {savepath}')

psds.to_pickle(savepath+'-psd.pkl')
cohs.to_pickle(savepath+'-coh.pkl')

