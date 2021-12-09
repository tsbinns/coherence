#### IMPORTS AND SETUP
import sys 
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('TKAgg')


### Gets dataset info
dataset_info = {'Berlin': {'main_path': 'C:\\Users\\tomth\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata',
                           'project_path': 'C:\\Users\\tomth\\OneDrive\\My Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\projects\\coherence',
                           'data': ['Rest-001-MedOff-StimOff', 'Rest-001-MedOn-StimOff',
                                    'Rest-002-MedOff-StimOff',
                                    'Rest-003-MedOff-StimOff', 'Rest-003-MedOn-StimOff',
                                    'Rest-004-MedOff-StimOff', 'Rest-004-MedOn-StimOff',
                                    'Rest-005-MedOff-StimOff', 'Rest-005-MedOn-StimOff',
                                    'Rest-006-MedOff-StimOff', 'Rest-006-MedOn-StimOff',
                                    'Rest-007-MedOff-StimOff', 'Rest-007-MedOn-StimOff',
                                    'Rest-008-MedOff-StimOff', 'Rest-008-MedOn-StimOff']},
                'Beijing': {'main_path': 'C:\\Users\\tomth\\Data\\BIDS_Beijing_ECOG_LFP\\rawdata',
                            'project_path': 'C:\\Users\\tomth\\OneDrive\\My Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Beijing_ECOG_LFP\\projects\\coherence',
                            'data': ['Rest-FOG006-MedOn-StimOff',
                                     'Rest-FOG008-MedOn-StimOff',
                                     'Rest-FOG010-MedOff-StimOff',
                                     'Rest-FOG011-MedOff-StimOff',
                                     'Rest-FOG012-MedOff-StimOff',
                                     'Rest-FOG014-MedOff-StimOff',
                                     'Rest-FOGC001-MedOff-StimOff']}}

## Commonly used sets of data
# Berlin & Beijing
berbei_all = {'Berlin': np.arange(len(dataset_info['Berlin']['data'])),
              'Beijing': np.arange(len(dataset_info['Beijing']['data']))} # Berlin & Beijing all data
berbei_OFF = {'Berlin': [0,2,3,5,7,9,11,13],
              'Beijing': [2,3,4,5,6]} # Berlin & Beijing medOff data
berbei_ON = {'Berlin': [1,4,6,8,10,12,14],
             'Beijing': [0,1]} # Berlin & Beijing medOn data
berbei_OFFandON = {'Berlin': [0,1,3,4,5,6,7,8,9,10,11,12,13,14], 
                   'Beijing': []} # Berlin & Beijing subjects with medOff & medOn data


# Berlin
ber_all = {'Berlin': np.arange(len(dataset_info['Berlin']['data']))} # All Berlin data
ber_OFF = {'Berlin': [0,2,3,5,7,9,11,13]} # Berlin medOff data
ber_ON = {'Berlin': [1,4,6,8,10,12,14]} # Berlin medOn data
ber_OFFandON = {'Berlin': [0,1,3,4,5,6,7,8,9,10,11,12,13,14]} # Berlin subjects with medOff & medOn data
ber_OFFandON_coords = {'Berlin': [0,1,3,4,5,6,7,8,9,10,11,12]} # Berlin subjects with medOff & medOn data with coords

# Beijing
bei_all = {'Beijing': np.arange(len(dataset_info['Beijing']['data']))} # All Beijing data
bei_OFF = {'Beijing': [2,3,4,5,6]} # Beijing medOff data
bei_ON = {'Beijing': [0,1]} # Beijing medOn data

### Gets path info
cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))

import plotting
from helpers import average_dataset, alter_by_condition, channel_reref_types, add_loc_groups


singlesubj_allchann = False # plots data for a single subject, averaged across runs
singlesubj_avgchann = False # plots data for a single subject, averaged across runs and channels
multiplesubj_allchann = False # plots data for multiple subjects, averaged across runs
multiplesubj_avgchann = True # plots data for multiple subjects, averaged across runs, channels, and subjects

subtract_med = True # subtracts the MedOn coherence from the MedOff coherence
subtract_baseline = False # subtracts the baseline coherence data from the real coherence data
group_by_region = True # groups channels that are being averaged based on the regions in which they are located

if subtract_med == True:
    mark_y0 = True


#### PLOTS DATA FOR A SINGLE SUBJECT, AVERAGED ACROSS RUNS
if singlesubj_allchann == True:
    ### Setup & Processing
    # Loads data
    datasets = []

    psds = []
    cohs = []
    for dataset in datasets.keys():
        project_path = dataset_info[dataset]['project_path']
        for data in datasets[dataset]:
            data_name = dataset_info[dataset]['data'][data]
            psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-psd.pkl")))
            cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-coh.pkl")))

            # Subtracts the baseline coherence from the genuine coherence, if requested
            if subtract_baseline == True:
                cohs[-1] = alter_by_condition(data=cohs[-1], cond='data_type', types=['real', 'shuffled'],
                                              method='subtract',
                                              separate=['run', 'ch_name_cortical', 'ch_name_deep', 'method'],
                                              x_keys=['subject', 'med', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                                      'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                                              y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

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

    psd = pd.concat(psds[:], ignore_index=True)
    coh = pd.concat(cohs[:], ignore_index=True)

    # Subtracts MedOn from MedOff data, if requested
    if subtract_med == True:
        psd = alter_by_condition(data=psd, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name', 'data_type', 'reref_type', 'ch_type'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands', 'ch_coords'],
                                 y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 recalculate_maxs=False, avg_as_equal=True)

        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method', 'reref_type_cortical', 'reref_type_deep'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical', 'ch_coords_deep'],
                                 y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 recalculate_maxs=False)


    ### Plotting
    # PSD
    plotting.psd_freqwise(psd, group_master=['ch_type'],
                          group_fig=['reref_type', 'ch_type'],
                          group_plot=['ch_name'],
                          plot_shuffled=False, plot_std=False, freq_limit=50, power_limit=15, mark_y0=mark_y0,
                          same_y=True)
    plotting.psd_bandwise(psd, group_master=['ch_type'],
                          group_fig=['reref_type', 'ch_type', 'data_type'],
                          group_plot=['ch_name'],
                          plot_shuffled=False, plot_std=False, same_y=True, plot_layout=[3,4])
    plotting.psd_bandwise_gb(psd, areas=['cortical'], group_master=['ch_type'],
                             group_fig=['reref_type', 'data_type'],
                             group_plot=['med'],
                             plot_shuffled=False, same_y_bandwise=True)

    # Coherence
    plotting.coh_freqwise(coh, group_master=['method'],
                          group_fig=['reref_type_cortical', 'reref_type_deep'],
                          group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep'],
                          plot_shuffled=False, plot_std=False, freq_limit=50, same_y=True, mark_y0=mark_y0,)
    plotting.coh_bandwise(coh, group_master=['method'],
                          group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'data_type'],
                          group_plot=['ch_name_cortical', 'ch_name_deep'],
                          plot_shuffled=False, plot_std=False, same_y=True, plot_layout=[3,4])
    plotting.coh_bandwise_gb(coh, areas=['cortical'], group_master=['method'],
                             group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'data_type'],
                             group_plot=['med'],
                             plot_shuffled=False, same_y_bandwise=True)


##### PLOTS DATA FOR A SINGLE SUBJECT, AVERAGED ACROSS RUNS AND CHANNELS
if singlesubj_avgchann == True:
    ### Setup & Processing
    # Loads data
    datasets = ber_OFFandON

    psds = []
    cohs = []
    for dataset in datasets.keys():
        project_path = dataset_info[dataset]['project_path']
        for data in datasets[dataset]:
            data_name = dataset_info[dataset]['data'][data]
            psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-psd.pkl")))
            cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-coh.pkl")))

            # Adds the channel localisation groups to the DataFrames
            ch_groups = 'all' # if data is not being grouped by region, set all channels to have the same region
            if group_by_region == True: # if data is being grouped by region, set the specific regions of each channel
                sub_id = np.unique(psds[-1]['subject'])
                if len(sub_id) != 1:
                    raise ValueError("Only the data of 1 subject should be present.")
                groups_fname = os.path.join(project_path, f"channel groups\\ch_groups-{sub_id[0]}.csv")
                ch_groups = pd.read_csv(groups_fname, delimiter=',') # localisation groups of the channels
            # Adds the groups
            psds[-1] = add_loc_groups(psds[-1], ch_groups, ch_name_key="ch_name", ch_coords_key="ch_coords")
            cohs[-1] = add_loc_groups(cohs[-1], ch_groups, ch_name_key="ch_name_cortical",
                                      ch_coords_key="ch_coords_cortical")


            # Subtracts the baseline coherence from the genuine coherence, if requested
            if subtract_baseline == True:
                cohs[-1] = alter_by_condition(data=cohs[-1], cond='data_type', types=['real', 'shuffled'],
                                              method='subtract',
                                              separate=['run', 'ch_name_cortical', 'ch_name_deep', 'method'],
                                              x_keys=['subject', 'med', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                                      'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep', 'loc_group'],
                                              y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

            # Averages data over runs
            psds[-1] = average_dataset(data=psds[-1], avg_over='run',
                                       separate=['ch_name', 'data_type', 'reref_type'],
                                       x_keys=['med', 'stim', 'task', 'subject', 'ch_type', 'freqs', 'fbands', 'ch_coords', 'loc_group'],
                                       y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

            cohs[-1] = average_dataset(data=cohs[-1], avg_over='run',
                                       separate=['ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                                       x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands', 'ch_coords_cortical',
                                               'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep', 'loc_group'],
                                       y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

            # Averages data over channels
            psds[-1] = average_dataset(data=psds[-1], avg_over='ch_name',
                                       separate=['ch_type', 'data_type', 'reref_type', 'loc_group'],
                                       x_keys=['med','stim','task','subject','freqs', 'fbands'],
                                       y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords'])

            cohs[-1] = average_dataset(data=cohs[-1], avg_over='ch_name_cortical',
                                       separate=['data_type', 'reref_type_cortical', 'reref_type_deep',
                                                 'method', 'loc_group'],
                                       x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands'],
                                       y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical',
                                               'ch_coords_deep'])

    psd = pd.concat(psds[:], ignore_index=True)
    coh = pd.concat(cohs[:], ignore_index=True)

    # Subtracts MedOn from MedOff data, if requested
    if subtract_med == True:
        psd = alter_by_condition(data=psd, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name', 'data_type', 'reref_type', 'ch_type', 'loc_group'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands', 'ch_coords'],
                                 y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 recalculate_maxs=False, avg_as_equal=True)

        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method', 'reref_type_cortical', 'reref_type_deep', 'loc_group'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical', 'ch_coords_deep'],
                                 y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 recalculate_maxs=False, avg_as_equal=True)


    ### Plotting
    # PSD
    plotting.psd_freqwise(psd, group_master=['ch_type'],
                          group_plot=['reref_type', 'ch_name', 'loc_group'],
                          plot_shuffled=False, plot_std=False, freq_limit=50, ylim_max=[-1,1],
                          same_y=True, mark_y0=mark_y0, add_avg=['subject', 'coloured'])
    plotting.psd_bandwise(psd, group_master=['ch_type'],
                          group_fig=['reref_type', 'ch_type', 'data_type'],
                          group_plot=['ch_name', 'group'],
                          plot_shuffled=False, plot_std=False, same_y=True)

    # Coherence
    plotting.coh_freqwise(coh, group_master=['method'],
                          group_plot=['reref_type_cortical', 'reref_type_deep', 'loc_group'],
                          plot_shuffled=False, plot_std=False, freq_limit=50, same_y=True,
                          mark_y0=mark_y0, add_avg=['subject', 'coloured'])
    plotting.coh_bandwise(coh, group_master=['method'],
                          group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'loc_group'],
                          plot_shuffled=False, plot_std=False, same_y=True)


##### PLOTS DATA FOR MULTIPLE SUBJECTS, AVERAGED ACROSS RUNS (BUT NOT CHANNELS OR SUBJECTS) ON SURFACE PLOTS
if multiplesubj_allchann == True:
    ### Setup & Processing

    # Loads data
    datasets = []

    psds = []
    cohs = []
    for dataset in datasets.keys():
        project_path = dataset_info[dataset]['project_path']
        for data in datasets[dataset]:
            data_name = dataset_info[dataset]['data'][data]
            psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-psd.pkl")))
            cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-coh.pkl")))

            # Subtracts the baseline coherence from the genuine coherence, if requested
            if subtract_baseline == True:
                cohs[-1] = alter_by_condition(data=cohs[-1], cond='data_type', types=['real', 'shuffled'],
                                              method='subtract',
                                              separate=['run', 'ch_name_cortical', 'ch_name_deep', 'method'],
                                              x_keys=['subject', 'med', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                                      'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                                              y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

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

    psd = pd.concat(psds[:], ignore_index=True)
    coh = pd.concat(cohs[:], ignore_index=True)

    # Subtracts MedOn from MedOff data, if requested
    if subtract_med == True:
        psd = alter_by_condition(data=psd, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name', 'data_type', 'reref_type'],
                                 x_keys=['run', 'stim', 'task', 'ch_type', 'freqs', 'fbands', 'ch_coords'],
                                 y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 avg_as_equal=True, recalculate_maxs=False)

        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                         'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                                 y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 avg_as_equal=True, recalculate_maxs=False)


    ### Plotting
    # PSD
    plotting.psd_bandwise_gb(psd, areas=['cortical'], group_master=['reref_type', 'ch_type'],
                             group_fig=['reref_type', 'ch_type', 'data_type'],
                             group_plot=['med'],
                             plot_shuffled=False, same_y_bandwise=True, normalise=[True, ['subject']])

    # Coherence
    plotting.coh_bandwise_gb(coh, areas=['cortical'], group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                             group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'data_type'],
                             group_plot=['med'],
                             plot_shuffled=False, same_y_bandwise=True, normalise=[True, ['subject']])


##### PLOTS DATA FOR MULTIPLE SUBJECTS, AVERAGED ACROSS RUNS, CHANNELS, AND SUBJECTS
if multiplesubj_avgchann == True:
    ### Setup & Processing
    # Loads data
    datasets = ber_OFFandON_coords

    psds = []
    cohs = []
    for dataset in datasets.keys():
        project_path = dataset_info[dataset]['project_path']
        for data in datasets[dataset]:
            data_name = dataset_info[dataset]['data'][data]
            psds.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-psd.pkl")))
            cohs.append(pd.read_pickle(os.path.join(project_path, 'derivatives', f"{data_name}-coh.pkl")))

            # Adds the channel localisation groups to the DataFrames
            ch_groups = 'all' # if data is not being grouped by region, set all channels to have the same region
            if group_by_region == True: # if data is being grouped by region, set the specific regions of each channel
                sub_id = np.unique(psds[-1]['subject'])
                if len(sub_id) != 1:
                    raise ValueError("Only the data of 1 subject should be present.")
                groups_fname = os.path.join(project_path, f"channel groups\\ch_groups-{sub_id[0]}.csv")
                ch_groups = pd.read_csv(groups_fname, delimiter=',') # localisation groups of the channels
            # Adds the groups
            psds[-1] = add_loc_groups(psds[-1], ch_groups, ch_name_key="ch_name", ch_coords_key="ch_coords")
            cohs[-1] = add_loc_groups(cohs[-1], ch_groups, ch_name_key="ch_name_cortical",
                                      ch_coords_key="ch_coords_cortical")



            # Subtracts the baseline coherence from the genuine coherence, if requested
            if subtract_baseline == True:
                cohs[-1] = alter_by_condition(data=cohs[-1], cond='data_type', types=['real', 'shuffled'],
                                              method='subtract',
                                              separate=['run', 'ch_name_cortical', 'ch_name_deep', 'method'],
                                              x_keys=['subject', 'med', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                                      'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep', 'loc_group'],
                                              y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

            # Averages data over runs
            psds[-1] = average_dataset(data=psds[-1], avg_over='run',
                                       separate=['ch_name', 'data_type', 'reref_type'],
                                       x_keys=['med', 'stim', 'task', 'subject', 'ch_type', 'freqs', 'fbands', 'ch_coords', 'loc_group'],
                                       y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

            cohs[-1] = average_dataset(data=cohs[-1], avg_over='run',
                                       separate=['ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                                       x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands', 'ch_coords_cortical',
                                               'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep', 'loc_group'],
                                       y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'])

            # Averages data over channels
            psds[-1] = average_dataset(data=psds[-1], avg_over='ch_name',
                                       separate=['ch_type', 'data_type', 'reref_type', 'loc_group'],
                                       x_keys=['med','stim','task','subject','freqs', 'fbands'],
                                       y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords'])

            cohs[-1] = average_dataset(data=cohs[-1], avg_over='ch_name_cortical',
                                       separate=['ch_name_deep', 'data_type', 'reref_type_cortical', 'reref_type_deep', 'method', 'loc_group'],
                                       x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands'],
                                       y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical',
                                               'ch_coords_deep'])

    psd = pd.concat(psds[:], ignore_index=True)
    coh = pd.concat(cohs[:], ignore_index=True)

    # Averages over subjects
    psd = average_dataset(data=psd, avg_over='subject',
                          separate=['ch_type', 'data_type', 'reref_type', 'med', 'loc_group'],
                          x_keys=['stim', 'task', 'freqs', 'fbands'],
                          y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords'])

    coh = average_dataset(data=coh, avg_over='subject',
                          separate=['data_type', 'reref_type_cortical', 'reref_type_deep', 'method', 'med', 'loc_group'],
                          x_keys=['stim', 'task', 'freqs', 'fbands'],
                          y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical',
                                  'ch_coords_deep'])

    # Subtracts MedOn from MedOff data, if requested
    if subtract_med == True:
        psd = alter_by_condition(data=psd, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name', 'data_type', 'reref_type', 'ch_type', 'loc_group'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands'],
                                 y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 avg_as_equal=True, recalculate_maxs=False)

        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                 separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method', 'data_type', 'reref_type_cortical', 'reref_type_deep', 'loc_group'],
                                 x_keys=['run', 'stim', 'task', 'freqs', 'fbands'],
                                 y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                 avg_as_equal=True, recalculate_maxs=False)


    ### Plotting
    # PSD
    plotting.psd_freqwise(psd, group_master=['ch_type', 'subject'],
                          group_plot=['reref_type', 'ch_name'],
                          plot_shuffled=False, plot_std=True, freq_limit=50, ylim_max=[-1,1],
                          same_y=False, mark_y0=mark_y0)
    plotting.psd_bandwise(psd, group_master=['ch_type', 'subject'],
                          group_fig=['ch_type', 'subject', 'data_type', 'reref_type'],
                          group_plot=['ch_name', 'loc_group'],
                          plot_shuffled=False, plot_std=True, same_y=False)

    # Coherence
    plotting.coh_freqwise(coh, group_master=['method', 'subject'],
                          group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep'],
                          plot_shuffled=False, plot_std=True, freq_limit=50, same_y=False,
                          mark_y0=mark_y0)
    plotting.coh_bandwise(coh, group_master=['method', 'subject'],
                          group_fig=['subject', 'data_type', 'reref_type_cortical', 'reref_type_deep'],
                          group_plot=['ch_name_cortical', 'ch_name_deep', 'loc_group'],
                          plot_shuffled=False, plot_std=True, same_y=False)


