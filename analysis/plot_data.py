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

### Gets path info
cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))

import plotting
from helpers import average_dataset, alter_by_condition


singlesubj_allchann = True # plots data for a single subject, averaged across runs
singlesubj_avgchann = False # plots data for a single subject, averaged across runs and channels
multiplesubj_allchann = False # plots data for multiple subjects, averaged across runs
multiplesubj_avgchann = False # plots data for multiple subjects, averaged across runs, channels, and subjects

subtract_med = False # subtracts the MedOn coherence from the MedOff coherence
subtract_baseline = False # subtracts the baseline coherence data from the real coherence data


#### PLOTS DATA FOR A SINGLE SUBJECT, AVERAGED ACROSS RUNS
if singlesubj_allchann == True:
    ### Setup & Processing
    # Loads data
    datasets = {'Berlin': [],
                'Beijing': [3]}

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
                                            y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                            ignore_runs=False)

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
                                separate=['ch_name', 'data_type', 'reref_type'],
                                x_keys=['subject', 'stim', 'task', 'ch_type', 'freqs', 'fbands', 'ch_coords'],
                                y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                ignore_runs=True)
        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                separate=['ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                                x_keys=['subject', 'stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                        'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                                y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                ignore_runs=True)


    ### Plotting
    # PSD
    plotting.psd_freqwise(psd, group_master=['reref_type', 'ch_type'],
                        group_plot=['ch_name'],
                        plot_shuffled=False, plot_std=False, freq_limit=50, power_limit=15, mark_y0=False,
                        plot_layout=[3,4])
    plotting.psd_bandwise(psd, group_master=['reref_type', 'ch_type'],
                        group_fig=['reref_type', 'ch_type', 'data_type'],
                        group_plot=['ch_name'],
                        plot_shuffled=False, plot_std=False, same_y=True, plot_layout=[3,4])
    plotting.psd_bandwise_gb(psd, areas=['cortical'], group_master=['reref_type', 'ch_type'],
                            group_fig=['reref_type', 'ch_type', 'data_type'],
                            group_plot=['med'],
                            plot_shuffled=False, same_y_bandwise=True)

    # Coherence
    plotting.coh_freqwise(coh, group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                        group_plot=['ch_name_cortical', 'ch_name_deep'],
                        plot_shuffled=False, plot_std=False, freq_limit=50, same_y=True, mark_y0=False,
                        plot_layout=[3,4])
    plotting.coh_bandwise(coh, group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                        group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'data_type'],
                        group_plot=['ch_name_cortical', 'ch_name_deep'],
                        plot_shuffled=False, plot_std=False, same_y=True, plot_layout=[3,4])
    plotting.coh_bandwise_gb(coh, areas=['cortical'], group_master=['reref_type_cortical', 'reref_type_deep', 'method'],
                            group_fig=['reref_type_cortical', 'reref_type_deep', 'method', 'data_type'],
                            group_plot=['med'],
                            plot_shuffled=False, same_y_bandwise=True)


##### PLOTS DATA FOR A SINGLE SUBJECT, AVERAGED ACROSS RUNS AND CHANNELS
if singlesubj_avgchann == True:
    ### Setup & Processing
    # Loads data
    datasets = {'Berlin': [13,14],
                'Beijing': []}

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
                                            y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                            ignore_runs=False)

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
                                    separate=['ch_name_deep', 'data_type', 'reref_type_cortical', 'reref_type_deep',
                                                'method'],
                                    x_keys=['med', 'stim', 'task', 'subject', 'freqs', 'fbands'],
                                    y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical',
                                            'ch_coords_deep'])

    psd = pd.concat(psds[:], ignore_index=True)
    coh = pd.concat(cohs[:], ignore_index=True)

    # Subtracts MedOn from MedOff data, if requested
    if subtract_med == True:
        psd = alter_by_condition(data=psd, cond='med', types=['Off', 'On'], method='subtract',
                                separate=['subject', 'ch_name', 'data_type', 'reref_type'],
                                x_keys=['stim', 'task', 'ch_type', 'freqs', 'fbands', 'ch_coords'],
                                y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                ignore_runs=True)
        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                                x_keys=['stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                        'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                                y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                ignore_runs=True)


    ### Plotting
    # PSD
    plotting.psd_freqwise(psd, group_master=['ch_type'],
                        group_plot=['reref_type', 'ch_name'],
                        plot_shuffled=False, plot_std=False, n_plots_per_page=6, freq_limit=50, power_limit=15,
                        same_y=False, mark_y0=False)
    plotting.psd_bandwise(psd, group_master=['ch_type'],
                        group_fig=['ch_type', 'data_type'],
                        group_plot=['reref_type', 'ch_name'],
                        plot_shuffled=False, plot_std=False, n_plots_per_page=6, same_y=False)

    # Coherence
    plotting.coh_freqwise(coh, group_master=['method'],
                        group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep'],
                        plot_shuffled=False, plot_std=False, n_plots_per_page=6, freq_limit=50, same_y=False,
                        mark_y0=False)
    plotting.coh_bandwise(coh, group_master=['method'],
                        group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep', 'data_type'],
                        plot_shuffled=False, plot_std=False, n_plots_per_page=6, same_y=False)


##### PLOTS DATA FOR MULTIPLE SUBJECTS, AVERAGED ACROSS RUNS (BUT NOT CHANNELS OR SUBJECTS) ON SURFACE PLOTS
if multiplesubj_allchann == True:
    ### Setup & Processing

    # Loads data
    datasets = {'Berlin': [0,1,3,4,5,6,7,8,9,10,11,12],
                'Beijing': []}

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
                                            y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                            ignore_runs=False)

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
                                x_keys=['stim', 'task', 'ch_type', 'freqs', 'fbands', 'ch_coords'],
                                y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                ignore_runs=True)
        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method'],
                                x_keys=['stim', 'task', 'freqs', 'fbands', 'ch_coords_cortical',
                                        'ch_coords_deep', 'reref_type_cortical', 'reref_type_deep'],
                                y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                ignore_runs=True)


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
    datasets = {'Berlin': [0,1,3,4,5,6,7,8,9,10,11,12,13,14],
                'Beijing': []}

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

    psd = pd.concat(psds[:], ignore_index=True)
    coh = pd.concat(cohs[:], ignore_index=True)

    # Averages over subjects
    psd = average_dataset(data=psd, avg_over='subject',
                        separate=['ch_type', 'data_type', 'reref_type', 'med'],
                        x_keys=['stim', 'task', 'freqs', 'fbands'],
                        y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords'])

    coh = average_dataset(data=coh, avg_over='subject',
                        separate=['data_type', 'reref_type_cortical', 'reref_type_deep', 'method', 'med'],
                        x_keys=['stim', 'task', 'freqs', 'fbands'],
                        y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax', 'ch_coords_cortical',
                                'ch_coords_deep'])

    # Subtracts MedOn from MedOff data, if requested
    if subtract_med == True:
        psd = alter_by_condition(data=psd, cond='med', types=['Off', 'On'], method='subtract',
                                separate=['subject', 'ch_name', 'data_type', 'reref_type', 'ch_type'],
                                x_keys=['run', 'stim', 'task', 'freqs', 'fbands'],
                                y_keys=['psd', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                avg_as_equal=True)
        coh = alter_by_condition(data=coh, cond='med', types=['Off', 'On'], method='subtract',
                                separate=['subject', 'ch_name_cortical', 'ch_name_deep', 'data_type', 'method', 'data_type', 'reref_type_cortical', 'reref_type_deep'],
                                x_keys=['run', 'stim', 'task', 'freqs', 'fbands'],
                                y_keys=['coh', 'fbands_avg', 'fbands_max', 'fbands_fmax'],
                                avg_as_equal=True)


    ### Plotting
    # PSD
    plotting.psd_freqwise(psd, group_master=['ch_type', 'subject'],
                        group_plot=['reref_type', 'ch_name'],
                        plot_shuffled=False, plot_std=True, n_plots_per_page=6, freq_limit=50, power_limit=15,
                        same_y=False, mark_y0=False)
    plotting.psd_bandwise(psd, group_master=['ch_type', 'subject'],
                        group_fig=['ch_type', 'subject', 'data_type'],
                        group_plot=['reref_type', 'ch_name'],
                        plot_shuffled=False, plot_std=True, n_plots_per_page=6, same_y=False)

    # Coherence
    plotting.coh_freqwise(coh, group_master=['method', 'subject'],
                        group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep'],
                        plot_shuffled=False, plot_std=True, n_plots_per_page=6, freq_limit=50, same_y=False,
                        mark_y0=False)
    plotting.coh_bandwise(coh, group_master=['method', 'subject'],
                        group_plot=['reref_type_cortical', 'reref_type_deep', 'ch_name_cortical', 'ch_name_deep', 'data_type'],
                        plot_shuffled=False, plot_std=True, n_plots_per_page=6, same_y=False)


