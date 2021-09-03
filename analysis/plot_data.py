#### IMPORTS AND SETUP
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

"""
#### PLOTS DATA FOR A SINGLE SUBJECT, AVERAGED ACROSS RUNS
### Setup & Processing
# Loads data
datasets = ['Rest-003-MedOff-StimOff', 'Rest-003-MedOn-StimOff']

psds = []
cohs = []
for data in datasets:
    psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-psd.pkl')))
    cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-coh.pkl')))

    # Averages data over runs
    psds[-1] = average_dataset(data=psds[-1], avg_over='run',
                               separate=['ch_name', 'data_type'],
                               x_keys=['med','stim','task','subject','ch_type','freqs'],
                               y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

    cohs[-1] = average_dataset(data=cohs[-1], avg_over='run',
                               separate=['ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                               x_keys=['med','stim','task','subject','freqs','fbands'],
                               y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

psd = pd.concat(psds[:], ignore_index=True)
coh = pd.concat(cohs[:], ignore_index=True)


### Plotting
# PSD
plotting.psd_freqwise(psd, group_master=['reref_type', 'ch_type'],
                      group_plot=['ch_name'],
                      plot_shuffled=False, plot_std=False, freq_limit=50, power_limit=15)
plotting.psd_bandwise(psd, group_master=['reref_type', 'ch_type'],
                      group_fig=['reref_type', 'ch_type', 'med', 'data_type'],
                      group_plot=['ch_name'],
                      plot_shuffled=False, plot_std=False, same_y=True)
plotting.psd_bandwise_gb(psd, areas=['cortical'], group_master=['reref_type', 'ch_type'],
                         group_fig=['reref_type', 'ch_type', 'med', 'data_type'],
                         plot_shuffled=False, same_y=True)

# Coherence
plotting.coh_freqwise(coh, group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                      group_plot=['ch_name_cortical', 'ch_name_deep'],
                      plot_shuffled=True, plot_std=False, freq_limit=50, same_y=True)
plotting.coh_bandwise(coh, group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                      group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'med', 'data_type'],
                      group_plot=['ch_name_cortical', 'ch_name_deep'],
                      plot_shuffled=False, plot_std=False, same_y=True)
plotting.coh_bandwise_gb(coh, areas=['cortical'], group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                         group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'med', 'data_type'],
                         plot_shuffled=False, same_y=True)
"""

"""
##### PLOTS DATA FOR MULTIPLE SUBJECTS, AVERAGED ACROSS RUNS, CHANNELS, AND (OPTIONALLY) SUBJECTS
### Setup & Processing
avg_subjects = True

# Loads data
datasets = ['Rest-001-MedOff-StimOff', 'Rest-001-MedOn-StimOff',
            'Rest-002-MedOff-StimOff', 
            'Rest-003-MedOff-StimOff', 'Rest-003-MedOn-StimOff',
            'Rest-004-MedOff-StimOff', 'Rest-004-MedOn-StimOff',
            'Rest-005-MedOff-StimOff', 'Rest-005-MedOn-StimOff']

psds = []
cohs = []
for set_i, data in enumerate(datasets):
    psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-psd.pkl')))
    cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-coh.pkl')))

    # Averages data over runs
    psds[-1] = average_dataset(data=psds[-1], avg_over='run',
                               separate=['ch_name', 'data_type', 'reref_type'],
                               x_keys=['med', 'stim', 'task', 'subject', 'ch_type', 'freqs', 'fbands', 'ch_coords'],
                               y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

    cohs[-1] = average_dataset(data=cohs[-1], avg_over='run',
                               separate=['ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                               x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands', 'ch_coords_cortical',
                                       'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                               y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

    # Averages data over channels
    psds[-1] = average_dataset(data=psds[-1], avg_over='ch_name',
                               separate=['ch_type', 'data_type', 'reref_type'],
                               x_keys=['med','stim','task','subject','freqs', 'fbands'],
                               y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords'])

    cohs[-1] = average_dataset(data=cohs[-1], avg_over='ch_name_cortical',
                               separate=['ch_name_deep', 'data_type', 'reref_type_cortical', 'reref_type_deep', 'method'],
                               x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands'],
                               y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical',
                                       'ch_coords_deep'])

    if set_i == len(datasets)-1:
        psd = pd.concat(psds[:], ignore_index=True)
        coh = pd.concat(cohs[:], ignore_index=True)

# Averages over subjects
if avg_subjects == True:
    psd = average_dataset(data=psd, avg_over='subject',
                          separate=['ch_type', 'data_type', 'reref_type', 'med'],
                          x_keys=['stim', 'task', 'freqs', 'fbands'],
                          y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords'])

    coh = average_dataset(data=coh, avg_over='subject',
                          separate=['data_type', 'reref_type_cortical', 'reref_type_deep', 'method', 'med'],
                          x_keys=['stim', 'task', 'freqs', 'fbands'],
                          y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical', 'ch_coords_deep'])


### Plotting
# PSD
plotting.psd_freqwise(psd, group_master=['reref_type', 'ch_type', 'subject'],
                      group_plot=['ch_name'],
                      plot_shuffled=False, plot_std=True, n_plots_per_page=1, freq_limit=50, power_limit=15)
plotting.psd_bandwise(psd, group_master=['reref_type', 'ch_type', 'subject'],
                      group_plot=['ch_name', 'med', 'data_type'],
                      plot_shuffled=False, plot_std=True, n_plots_per_page=2)

# Coherence
plotting.coh_freqwise(coh, group_master=['reref_type_cortical', 'reref_type_deep', 'method', 'subject'],
                            group_plot=['ch_name_cortical', 'ch_name_deep'],
                            plot_shuffled=True, plot_std=True, n_plots_per_page=1, freq_limit=50, same_y=True)
plotting.coh_bandwise(coh, group_master=['reref_type_cortical', 'reref_type_deep', 'method', 'subject'],
                            group_plot=['ch_name_cortical', 'ch_name_deep', 'med', 'data_type'],
                            plot_shuffled=False, plot_std=True, n_plots_per_page=2)
"""

""""""
##### PLOTS DATA FOR MULTIPLE SUBJECTS, AVERAGED ACROSS RUNS (BUT NOT CHANNELS OR SUBJECTS) ON SURFACE PLOTS
### Setup & Processing
# Loads data
datasets = ['Rest-001-MedOff-StimOff', 'Rest-001-MedOn-StimOff',
            'Rest-002-MedOff-StimOff', 
            'Rest-003-MedOff-StimOff', 'Rest-003-MedOn-StimOff',
            'Rest-004-MedOff-StimOff', 'Rest-004-MedOn-StimOff',
            'Rest-005-MedOff-StimOff', 'Rest-005-MedOn-StimOff']

psds = []
cohs = []
for set_i, data in enumerate(datasets):
    psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-psd.pkl')))
    cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f'{data}-coh.pkl')))

    # Averages data over runs
    psds[-1] = average_dataset(data=psds[-1], avg_over='run',
                               separate=['ch_name', 'data_type', 'reref_type'],
                               x_keys=['med', 'stim', 'task', 'subject', 'ch_type', 'freqs', 'fbands', 'ch_coords'],
                               y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

    cohs[-1] = average_dataset(data=cohs[-1], avg_over='run',
                               separate=['ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                               x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands', 'ch_coords_cortical',
                                       'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                               y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

    if set_i == len(datasets)-1:
        psd = pd.concat(psds[:], ignore_index=True)
        coh = pd.concat(cohs[:], ignore_index=True)


### Plotting
# PSD
plotting.psd_bandwise_gb(psd, areas=['cortical'], group_master=['reref_type', 'ch_type'],
                         group_fig=['reref_type', 'ch_type', 'med', 'data_type'],
                         plot_shuffled=False, same_y=True)

# Coherence
plotting.coh_bandwise_gb(coh, areas=['cortical'], group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                         group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'med', 'data_type'],
                         plot_shuffled=False, same_y=True)
""""""