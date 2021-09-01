from operator import methodcaller
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import full
import helpers


def psd(psd, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, n_plots_per_page=6, freq_limit=None,
        power_limit=None, same_y=True):
    """ Plots PSDs of the data.

    PARAMETERS
    ----------
    psds : pandas DataFrame
    -   A DataFrame containing the channel names, their types, and the normalised power values.

    group_master : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate data into groups that share the
        same y-axis limits (if applicable)

    group_fig : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate the grouped data into
        subgroups used for plotting the same figure(s) with the same title. If empty, the same groups as group_master
        are used.

    group_plot : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate the subgrouped data (specified
        by group_figure) into further subgroups used for plotting data on the same plot. If empty, the same groups as
        group_fig are used.

    plot_shuffled : bool, default False
    -   Whether or not to plot PSDs for the shuffled LFP data.

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.

    freq_limit : int | float
    -   The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.

    power_limit : int | float
    -   The y-axis limit for PSDs. None by default (i.e. no limit).

    same_y : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True (default), the same axes are
        used; if False, the same axes are not used.


    RETURNS
    ----------
    N/A
    
    """

    ### Setup
    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Establishes keys for groups used in generating labels for data
    psd_data_keys = ['freqs', 'psd', 'psd_std'] # keys containing data that do not represent different conditions
    fig_keys = [key for key in psd.keys() if key not in psd_data_keys and key not in group_plot]
    plot_keys = [key for key in psd.keys() if key not in psd_data_keys and key not in group_fig]
    data_keys = [key for key in psd.keys() if key not in psd_data_keys and key not in group_fig]

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(psd.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)

    # Gets indices of master-grouped data
    names_master = helpers.combine_names(psd, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)
    
    # Gets colours of data
    colour_info = helpers.get_colour_info(psd, [key for key in psd.keys() if key not in psd_data_keys
                                                and key not in group_master and key not in group_fig
                                                and key not in group_plot])
    colours = helpers.data_colour(colour_info, not_for_unique=True, avg_as_equal=True)


    ### Plotting
    for mastergroup_i, idc_group_master in enumerate(idcs_group_master):
        
        # Gets a global y-axis for all data of the same mastergroup (if requested)
        if same_y == True:
            if plot_std == True and 'psd_std' in psd.keys():
                group_ylim = helpers.same_axes(psd.psd[idc_group_master] + psd.psd_std[idc_group_master])
            else:
                group_ylim = helpers.same_axes(psd.psd[idc_group_master])
        group_ylim[0] = 0

        # Gets indices of figure-grouped data
        names_fig = helpers.combine_names(psd.iloc[idc_group_master], group_fig, joining=',')
        names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
        for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]


        for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
            
            # Gets the characteristics for all data of this type so that a title for the window can be generated
            fig_info = {}
            for key in fig_keys:
                fig_info[key] = list(np.unique(psd[key][idc_group_fig]))
            wind_title, included = helpers.window_title(fig_info, base_title='PSD:', full_info=False)

            names_plot = helpers.combine_names(psd, group_plot, joining=',')
            names_group_plot, idcs_group_plot = helpers.unique_names([names_plot[i] for i in idc_group_fig])
            for plotgroup_i, idc_group_plot in enumerate(idcs_group_plot):
                idcs_group_plot[plotgroup_i] = [idc_group_fig[i] for i in idc_group_plot]

            n_plots = len(idcs_group_plot) # number of plots to make for this type
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
            n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

            stop = False
            plotgroup_i = 0
            for page_i in range(n_pages): # for each page of this type

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_plots_per_page == 1:
                    axs = np.asarray([[axs]])
                plt.tight_layout(rect = [0, 0, 1, .97])
                fig.suptitle(wind_title)

                for row_i in range(n_rows): # fill up each row from top to down...
                    for col_i in range(n_cols): # ... and from left to right
                            if stop is False: # if there is still data to plot for this subgroup

                                idc_group_plot = idcs_group_plot[plotgroup_i]

                                # Gets the characteristics for all data of this plot so that a title can be generated
                                plot_info = {}
                                for key in plot_keys:
                                    plot_info[key] = list(np.unique(psd[key][idc_group_plot]))
                                plot_title, plot_included = helpers.plot_title(plot_info, already_included=included,
                                                                               full_info=False)
                                included.extend(plot_included) # keeps track of what is already included in the titles

                                # Sets up subplot
                                axs[row_i, col_i].set_title(plot_title)
                                axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                                axs[row_i, col_i].set_ylabel('Normalised Power (% total)')

                                for ch_idx in idc_group_plot: # for each data entry
                                
                                    data = psd.iloc[ch_idx] # the data to plot

                                    if freq_limit != None: # finds limit of frequencies to plot (if applicable)...
                                        freq_limit_i = int(np.where(data.freqs == data.freqs[data.freqs >=
                                                    freq_limit].min())[0])
                                    else: #... or plots all data
                                        freq_limit_i = len(data.freqs)

                                    # Gets the characteristics for the data so that a label can be generated
                                    data_info = {}
                                    for key in data_keys:
                                        data_info[key] = data[key]
                                    data_title = helpers.data_title(data_info, already_included=included, full_info=False)
                                    if data_title == '': # don't label the data if there is no info to add
                                        data_title = None

                                    # Gets the colour of data based on it's characteristics
                                    colour = []
                                    for key in colours.keys():
                                        colour.append(colours[key][colour_info[key][1].index(data[key])])
                                    if colour:
                                        colour = np.nanmean(colour, axis=0) # takes the average colour based on the data's...
                                        #... characteristics
                                    else: # if there is no colour info, set the colour to black
                                        colour = [0, 0, 0, 1]
                                    
                                    # Plots data
                                    axs[row_i, col_i].plot(data.freqs[:freq_limit_i+1], data.psd[:freq_limit_i+1],
                                                        label=data_title, linewidth=2, color=colour)
                                    if data_title != None: # if data has been labelled, plot the legend
                                        axs[row_i, col_i].legend()
                                    
                                    # Plots std (if applicable)
                                    if plot_std == True:
                                        if 'psd_std' in data.keys():
                                            std_plus = data.psd[:freq_limit_i+1] + data.psd_std[:freq_limit_i+1]
                                            std_minus = data.psd[:freq_limit_i+1] - data.psd_std[:freq_limit_i+1]
                                            axs[row_i, col_i].fill_between(data.freqs[:freq_limit_i+1], std_plus, std_minus,
                                                                        color=colour, alpha=colour[-1]*.2)

                                    # Sets all y-axes to be equal (if requested)
                                    if same_y == True:
                                        axs[row_i, col_i].set_ylim(*group_ylim)

                                    # Cuts off power (if requested)
                                    if power_limit != None:
                                        ylim = axs[row_i, col_i].get_ylim()
                                        if ylim[1] > power_limit:
                                            axs[row_i, col_i].set_ylim(ylim[0], power_limit)

                                plotgroup_i += 1
                                if ch_idx == idc_group_plot[-1]: # if there is no more data to plot for this type...
                                    stop = True #... don't plot anything else
                                    extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                    #... be removed

                            elif stop is True and extra > 0: # if there is no more data to plot for this type...
                                fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()



def coherence_freqwise(coh, separate_top, separate_sub=None, plot_shuffled=False, plot_std=True, n_plots_per_page=6,
                       freq_limit=None, methods=['coh', 'imcoh'], same_y=True):
    """ Plots single-frequency-wise coherence data.

    PARAMETERS
    ----------
    coh : pandas DataFrame
    -   A DataFrame containing the corresponding ECoG and LFP channel names, and the single-frequency-wise coherence
        data.

    separate_top : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate data into groups.

    separate_sub : list of strs | None
    -   Keys of coh containing the data characteristics which should be used to separate the grouped data into
        subgroups.

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_std : bool, default True
    -   Whether or not to plot standard deviation values (if they are present) alongside the data.

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.

    freq_limit : int | float
    -   The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.

    methods : list of str
    -   The methods used to calculate coherence. By default, 'coh' (standard coherence) and 'imcoh' (imaginary
        coherence)


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)
    
    # Gets indices of grouped data
    names_top = helpers.combine_names(coh, separate_top, joining=',')
    group_names_top, group_idcs_top = helpers.unique_names(names_top)

    # Gets colours of data
    colour_info = {
        'med': ['binary', list(np.unique(coh.med))],
        'stim': ['binary', list(np.unique(coh.stim))],
        'task': ['non-binary', list(np.unique(coh.task))],
        'subject': ['non-binary', list(np.unique(coh.subject))],
        'run': ['non-binary', list(np.unique(coh.run))]
    }
    colours = helpers.data_colour(colour_info, not_for_unique=True, avg_as_equal=True)


    ### Plotting
    for method in methods:

        for topgroup_i, group_idc_top in enumerate(group_idcs_top):
            # Gets the characteristics for all the data so that a title for the window can be generated
            group_info = {
                'med': list(np.unique(coh.med[group_idc_top])),
                'stim': list(np.unique(coh.stim[group_idc_top])),
                'task': list(np.unique(coh.task[group_idc_top])),
                'subject': list(np.unique(coh.subject[group_idc_top])),
                'run': list(np.unique(coh.run[group_idc_top]))
            }
            for key in separate_top:
                group_info[key] = list(np.unique(coh[key][group_idc_top]))
            wind_title, included = helpers.window_title(group_info, base_title=f'{method}:', full_info=False)

            # Gets a global y-axis for all data of the same type (if requested)
            if same_y == True:
                if plot_std == True and f'{method}_std' in coh.keys():
                    ylim = []
                    ylim.extend(helpers.same_axes(coh[method][group_idc_top] + coh[f'{method}_std'][group_idc_top]))
                    ylim.extend(helpers.same_axes(coh[method][group_idc_top] - coh[f'{method}_std'][group_idc_top]))
                    group_ylim = [min(ylim), max(ylim)]
                else:
                    group_ylim = helpers.same_axes(coh[method][group_idc_top])

            if separate_sub != None:
                names_sub = helpers.combine_names(coh, separate_sub, joining=',')
                group_names_sub, group_idcs_sub = helpers.unique_names([names_sub[i] for i in group_idc_top])
                for subgroup_i, group_idc_sub in enumerate(group_idcs_sub):
                    group_idcs_sub[subgroup_i] = [group_idc_top[i] for i in group_idc_sub]
            else:
                group_names_sub = [group_names_top[topgroup_i]]
                group_idcs_sub = [group_idc_top]

            n_plots = len(group_idcs_sub) # number of plots to make for this type
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
            n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

            stop = False
            subgroup_i = 0
            for page_i in range(n_pages): # for each page of this type

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_plots_per_page == 1:
                    axs = np.asarray([[axs]])
                plt.tight_layout(rect = [0, 0, 1, .97])
                fig.suptitle(wind_title)

                for row_i in range(n_rows): # fill up each row from top to down...
                    for col_i in range(n_cols): # ... and from left to right
                        if stop is False: # if there is still data to plot for this type

                            group_idc_sub = group_idcs_sub[subgroup_i] # indices of the data entries of this subgroup

                            # Gets the characteristics for all data of this plot so that a title can be generated
                            plot_info = {
                                    'med': list(np.unique(coh.med[group_idc_sub])),
                                    'stim': list(np.unique(coh.stim[group_idc_sub])),
                                    'task': list(np.unique(coh.task[group_idc_sub])),
                                    'subject': list(np.unique(coh.subject[group_idc_sub])),
                                    'run': list(np.unique(coh.run[group_idc_sub])),
                                    ':': list(np.unique(coh.ch_name_deep[group_idc_sub]))
                                }
                            for key in separate_sub:
                                plot_info[key] = list(np.unique(coh[key][group_idc_sub]))
                            plot_title, plot_included = helpers.plot_title(plot_info, already_included=included,
                                                                           full_info=False)
                            included.append(plot_included)

                            # Sets up subplot
                            axs[row_i, col_i].set_title(plot_title)
                            axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                            axs[row_i, col_i].set_ylabel('Coherence')

                            for ch_idx in group_idc_sub: # for each data entry
                            
                                data = coh.iloc[ch_idx] # the data to plot

                                if freq_limit != None: # finds limit of frequencies to plot (if applicable)...
                                    freq_limit_i = int(np.where(data.freqs == data.freqs[data.freqs >=
                                                    freq_limit].min())[0])
                                else: #... or plots all data
                                    freq_limit_i = len(data.freqs)

                                # Gets the characteristics for the data so that a label can be generated
                                data_info = {
                                    'med': data.med,
                                    'stim': data.stim,
                                    'task': data.task,
                                    'subject': data.subject,
                                    'run': data.run,
                                    'ch': data.ch_name_deep
                                }
                                data_title = helpers.data_title(data_info, already_included=included, full_info=False)
                                if data_title == '': # don't label the data if there is no info to add
                                    data_title = None

                                # Gets the colour of data based on it's characteristics
                                colour = []
                                for key in colours.keys():
                                    colour.append(colours[key][colour_info[key][1].index(data[key])])
                                if colour:
                                    colour = np.nanmean(colour, axis=0)[0] # takes the average colour based on the data's...
                                    #... characteristics
                                else: # if there is no colour info, set the colour to black
                                    colour = [0, 0, 0]
                                
                                # Plots data
                                alpha = 1
                                if data.ch_name_deep[:8] == 'SHUFFLED':
                                    alpha /= 2
                                axs[row_i, col_i].plot(data.freqs[:freq_limit_i+1], data[method][:freq_limit_i+1],
                                                        label=data_title, linewidth=2, color=colour, alpha=alpha)
                                if data_title != None: # if data has been labelled, plot the legend
                                    axs[row_i, col_i].legend()
                                
                                # Plots std (if applicable)
                                if plot_std == True:
                                    std_name = f'{method}_std'
                                    if std_name in data.keys():
                                        std_plus = data[method][:freq_limit_i+1] + data[std_name][:freq_limit_i+1]
                                        std_minus = data[method][:freq_limit_i+1] - data[std_name][:freq_limit_i+1]
                                        axs[row_i, col_i].fill_between(data.freqs[:freq_limit_i+1], std_plus, std_minus,
                                                                        color=colour, alpha=alpha*.3)
                                
                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(*group_ylim)

                            subgroup_i += 1 # moves on to the next data to plot
                            if ch_idx == group_idc_sub[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                #... be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()



def coherence_bandwise(coh, separate_top, separate_sub=None, plot_shuffled=False, plot_std=True, n_plots_per_page=6, methods=['coh', 'imcoh'],
                       keys_to_plot=['avg', 'max'], same_y = True):
    """ Plots frequency band-wise coherence data.

    PARAMETERS
    ----------
    coh : pandas DataFrame
    -   A DataFrame containing the corresponding ECoG and LFP channel names, and the frequency band-wise coherence
        data.

    separate_top : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate data into groups.

    separate_sub : list of strs | None
    -   Keys of coh containing the data characteristics which should be used to separate the grouped data into
        subgroups.

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_std : bool, default True
    -   Whether or not to plot standard deviation values (if they are present) alongside the data.

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.

    methods : list of strs
    -   The methods used to calculate coherence. By default, 'coh' (standard coherence) and 'imcoh' (imaginary
        coherence).
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)
    
    # Gets indices of grouped data
    names_top = helpers.combine_names(coh, separate_top, joining=',')
    group_names_top, group_idcs_top = helpers.unique_names(names_top)

    # Gets the characteristics for all the data so that a title for the window can be generated
    dataset_info = {
        'med': list(np.unique(coh.med)),
        'stim': list(np.unique(coh.stim)),
        'task': list(np.unique(coh.task)),
        'subject': list(np.unique(coh.subject)),
        'run': list(np.unique(coh.run))
    }


    ## Plotting
    for method in methods: # plots data for different coherence calculations separately
        
        # Generates key names for the band-wise data based on the requested features in keys_to_plot. Follows the...
        #... form method_fbands_feature (e.g. coh_fbands_avg)
        fullkeys_to_plot = []
        for key in keys_to_plot:
            fullkeys_to_plot.append(f'{method}_fbands_{key}')

        for topgroup_i, group_idc_top in enumerate(group_idcs_top):
            
            # Gets the characteristics for all the data so that a title for the window can be generated
            group_info = {
                'med': list(np.unique(coh.med[group_idc_top])),
                'stim': list(np.unique(coh.stim[group_idc_top])),
                'task': list(np.unique(coh.task[group_idc_top])),
                'subject': list(np.unique(coh.subject[group_idc_top])),
                'run': list(np.unique(coh.run[group_idc_top]))
            }
            for key in separate_top:
                group_info[key] = list(np.unique(coh[key][group_idc_top]))
            wind_title, included = helpers.window_title(group_info, base_title=f'{method}:', full_info=False)

            # Gets a global y-axis for all data of the same type (if requested)
            if same_y == True:
                if plot_std == True and f'{method}_std' in coh.keys():
                    ylim = []
                    ylim.extend(helpers.same_axes(coh[[key for key in fullkeys_to_plot]].iloc[group_idc_top] + 
                                                  coh[[key+'_std' for key in fullkeys_to_plot]].iloc[group_idc_top]))
                    ylim.extend(helpers.same_axes(coh[[key for key in fullkeys_to_plot]].iloc[group_idc_top] - 
                                                  coh[[key+'_std' for key in fullkeys_to_plot]].iloc[group_idc_top]))
                    group_ylim = [min(ylim), max(ylim)]
                else:
                    group_ylim = helpers.same_axes(coh[[key for key in fullkeys_to_plot]].iloc[group_idc_top])

            if separate_sub != None:
                names_sub = helpers.combine_names(coh, separate_sub, joining=',')
                group_names_sub, group_idcs_sub = helpers.unique_names([names_sub[i] for i in group_idc_top])
                for subgroup_i, group_idc_sub in enumerate(group_idcs_sub):
                    group_idcs_sub[subgroup_i] = [group_idc_top[i] for i in group_idc_sub]
            else:
                group_names_sub = [group_names_top[topgroup_i]]
                group_idcs_sub = [group_idc_top]

            n_plots = len(group_idcs_sub) # number of plots to make for this type
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
            n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

            stop = False
            subgroup_i = 0
            for page_i in range(n_pages):

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_plots_per_page == 1:
                    axs = np.asarray([[axs]])
                plt.tight_layout(rect=[0, 0, 1, .95])
                fig.suptitle(wind_title)

                for row_i in range(n_rows): # fill up each row from top to down...
                    for col_i in range(n_cols): # ... and from left to right
                        if stop is False: # if there is still data to plot for this method
                            
                            group_idc_sub = group_idcs_sub[subgroup_i] # indices of the data entries of this subgroup

                            # Gets the characteristics for all data of this channel so that a title for the subplot can...
                            #... be generated
                            channel_info = {
                                'med': [data.med],
                                'stim': [data.stim],
                                'task': [data.task],
                                'subject': [data.subject],
                                'run': [data.run],
                            }
                            ch_title, _ = helpers.channel_title(channel_info, already_included=included,
                                                                base_title=f'{comb_ch_names[type_idx]},',
                                                                full_info=False)

                            # Sets up subplot
                            axs[row_i, col_i].set_title(ch_title)
                            axs[row_i, col_i].set_ylabel('Coherence')
                                
                            # Sets up data for plotting as bars
                            n_groups = len(keys_to_plot)
                            bands = data.fbands
                            n_bars = len(bands)
                            width = 1/n_bars

                            # Location of bars in the groups
                            start_locs = np.arange(n_groups, step=width*(n_bars+2)) # makes sure the bars of each group...
                            #... don't overlap
                            group_locs = []
                            for start_loc in start_locs: # x-axis bar positions, grouped by group
                                group_locs.append([start_loc+width*i for i in np.arange(n_bars)])
                            bar_locs = []
                            for bar_i in range(n_bars): # x-axis bar positions, grouped by band
                                bar_locs.append([])
                                for group_i in range(n_groups):
                                    bar_locs[bar_i].append(group_locs[group_i][bar_i])

                            # Gets the data to plot
                            if 'max' in keys_to_plot:
                                fmaxs = []

                            for band_i, band in enumerate(bands): # for each frequency band

                                to_plot = []
                                if plot_std == True:
                                    stds = []

                                for key in fullkeys_to_plot:
                                    to_plot.append(data[key][band_i]) # gets the data to be plotted...
                                    if plot_std == True: #... and the std of this data (if applicable)
                                        if f'{key}_std' in data.keys():
                                            stds.append(data[f'{key}_std'][band_i])
                                        else:
                                            stds.append(np.nan)

                                    if 'fbands_max' in key: # gets the std of the fmax data to add to the plots
                                        fmaxs.append(str(int(data[method+'_fbands_fmax'][band_i])))
                                        if plot_std == True and method+'_fbands_fmax_std' in data.keys():
                                            fmax_std = u'\u00B1'+str(int(np.ceil(data[method+'_fbands_fmax_std'][band_i])))
                                            fmaxs[-1] += fmax_std

                                # Plots the data
                                axs[row_i, col_i].bar(bar_locs[band_i], to_plot, width=width, label=band, alpha=.7)
                                if plot_std == True:
                                    axs[row_i, col_i].errorbar(bar_locs[band_i], to_plot, yerr=stds, capsize=3, fmt=' ')

                            # Sets all y-axes to be equal (if requested)
                            if same_y == True:
                                axs[row_i, col_i].set_ylim(ylim[0], ylim[1])

                            # Tidies up the x-axis ticks and labels
                            axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                            axs[row_i, col_i].set_xticklabels(keys_to_plot)
                            axs[row_i, col_i].legend()

                            # Adds the fmax data to the bars (if applicable)
                            if 'max' in keys_to_plot:
                                ylim = axs[row_i, col_i].get_ylim()
                                for fmax_i, fmax in enumerate(fmaxs):
                                    text_ypos = data[method+'_fbands_max'][fmax_i]+ylim[1]*.01 # where to put text
                                    if plot_std == True:
                                        text_ypos += data[method+'_fbands_max_std'][fmax_i]
                                    # adds the fmax values at an angle to the bars one at a time
                                    axs[row_i, col_i].text(group_locs[1][fmax_i], text_ypos, fmax, ha='center', rotation=60)
                                axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*.07]) # increases the subplot height...
                                #... to accomodate the text

                            ch_i += 1 # moves on to the next data to plot
                            if ch_i == len(type_idc): # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                #... be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()

