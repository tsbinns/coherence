import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import io
from scipy import stats
import pandas as pd
import datetime
import helpers


def psd_freqwise(psd, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, plot_layout=[2,3],
                 freq_limit=None, power_limit=None, same_y=True, avg_as_equal=True, mark_y0=False):
    """ Plots frequency-wise PSDs of the data.

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

    plot_layout : list of ints of size [1 x 2]
    -   A list specifying how plots should be layed out on the page. plot_layout[0] is the number of rows, and
        plot_layout[1] is the number of columns. [2, 3] by default, i.e. 2 rows and 3 columns.

    freq_limit : int | float
    -   The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.

    power_limit : int | float
    -   The y-axis limit for PSDs. None by default (i.e. no limit).

    same_y : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True (default), the same axes are
        used; if False, the same axes are not used.
    
    avg_as_equal : bool, default True
    -   Whether or not to treat averaged data as equivalent, regardless of what was averaged over. E.g. if some data had
        been averaged across subjects 1 & 2, but other data across subjects 3 & 4, avg_as_equal = True would treat this
        data as if it was the same group of data.

    mark_y0 : bool, default False
    -   Whether or not to draw a dotted line where y = 0. If False (default), no line is drawn. If True, a line is
        drawn.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Checks to make sure that S.D. data is present if it is requested
    if plot_std is True:
        std_present = False
        for col in psd.columns:
            if 'std' in col:
                std_present = True
        if std_present == False:
            print(f"Warning: Standard deviation data is not present, so it cannot be plotted.")
            plot_std = False

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(psd.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)

    # Sets averaged data to the same type, if requested
    if avg_as_equal == True:
        for key in psd.keys():
            for data_i, data in enumerate(psd[key]):
                if type(data) == str or type(data) == np.str_:
                    if data[:3] == 'avg':
                        psd[key].iloc[data_i] = 'avg'

    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Keys used to label figures, plots, and data
    psd_data_keys = ['ch_coords', 'ch_coords_std', 'freqs', 'psd', 'psd_std', 'fbands', 'fbands_avg', 'fbands_avg_std',
                     'fbands_max', 'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    data_keys = [key for key in psd.keys() if key not in group_fig+group_plot+psd_data_keys]
    plot_keys = [key for key in psd.keys() if key not in group_fig+psd_data_keys+data_keys]
    fig_keys = [key for key in psd.keys() if key not in psd_data_keys+data_keys+plot_keys]

    ## Alters level of keys depending on the values in the data
    # Moves eligible labels from data to subplot level
    move_up = []
    for key in data_keys:
        if len(np.unique(psd[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    data_keys = [key for key in data_keys if key not in move_up]
    plot_keys += move_up
    plot_keys = pd.unique(plot_keys).tolist()
    # Moves eligible labels from subplot to figure level
    move_up = []
    for key in plot_keys:
        if len(np.unique(psd[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    plot_keys = [key for key in plot_keys if key not in move_up]
    fig_keys += move_up
    fig_keys = pd.unique(fig_keys).tolist()

    # Gets indices of master-grouped data
    names_master = helpers.combine_names(psd, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)
    
    # Gets colours of data
    colour_info = helpers.get_colour_info(psd, [key for key in psd.keys() if key not in psd_data_keys
                                                and key not in group_master and key not in group_fig
                                                and key not in group_plot])
    colours = helpers.data_colour(colour_info, not_for_unique=True, avg_as_equal=avg_as_equal)

    # Name of the folder in which to save figures (based on the current time)
    foldername = 'psd_freqwise-'+''.join([str(x) for x in datetime.datetime.now().timetuple()[:-3]])


    ### Plotting
    for mastergroup_i, idc_group_master in enumerate(idcs_group_master):
        
        # Gets a global y-axis for all data of the same mastergroup (if requested)
        if same_y == True:
            if plot_std == True and 'psd_std' in psd.keys():
                group_ylim = helpers.same_axes(psd.psd[idc_group_master] + psd.psd_std[idc_group_master])
            else:
                group_ylim = helpers.same_axes(psd.psd[idc_group_master])
            #group_ylim[0] = 0

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
            n_plots_per_page = int(plot_layout[0]*plot_layout[1]) # number of plots on each page
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = plot_layout[0] # number of rows these pages will need
            n_cols = plot_layout[1] # number of columns these pages will need

            stop = False
            plotgroup_i = 0
            for page_i in range(n_pages): # for each page of this type

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_rows == 1 and n_cols == 1:
                    axs = np.asarray([[axs]]) # puts the lone axs into an array for later indexing
                elif n_rows == 1 and n_cols > 1:
                    axs = np.vstack((axs, [0,0])) # adds an extra row for later indexing
                elif n_cols == 1 and n_rows > 1:
                    axs = np.hstack((axs), [0,0]) # adds and extra column for later indexing
                plt.tight_layout(rect = [0, 0, 1, .95])
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
                            now_included = included + plot_included # keeps track of what is already included in the...
                            #... titles

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
                                data_title = helpers.data_title(data_info, already_included=now_included,
                                                                full_info=False)
                                if data_title == '': # don't label the data if there is no info to add
                                    data_title = None

                                # Gets the colour of data based on it's characteristics
                                colour = []
                                for key in colours.keys():
                                    colour.append(colours[key][colour_info[key][1].index(data[key])])
                                if colour:
                                    colour = np.nanmean(colour, axis=0) # takes the average colour based on the...
                                    #... data's characteristics
                                else: # if there is no colour info, set the colour to black
                                    colour = [0, 0, 0, 1]
                                
                                # Plots data
                                axs[row_i, col_i].plot(data.freqs[:freq_limit_i+1], data.psd[:freq_limit_i+1],
                                                    label=data_title, linewidth=2, color=colour)
                                if data_title != None: # if data has been labelled, plot the legend
                                    axs[row_i, col_i].legend(labelspacing=0)
                                
                                # Plots std (if applicable)
                                if plot_std == True:
                                    if 'psd_std' in data.keys():
                                        std_plus = data.psd[:freq_limit_i+1] + data.psd_std[:freq_limit_i+1]
                                        std_minus = data.psd[:freq_limit_i+1] - data.psd_std[:freq_limit_i+1]
                                        axs[row_i, col_i].fill_between(data.freqs[:freq_limit_i+1], std_plus, std_minus,
                                                                       color=colour, alpha=colour[-1]*.2)

                                # Demarcates 0 on the y-axis, if requested
                                if mark_y0 == True:
                                    xlim = axs[row_i, col_i].get_xlim()
                                    xlim_trim = (xlim[1] - xlim[0])*.05
                                    axs[row_i, col_i].plot([xlim[0]+xlim_trim, xlim[1]-xlim_trim], [0,0], color='grey',
                                        linestyle='dashed')

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(*group_ylim)

                                # Cuts off power (if requested)
                                if power_limit != None:
                                    ylim = axs[row_i, col_i].get_ylim()
                                    if ylim[1] > power_limit:
                                        axs[row_i, col_i].set_ylim(ylim[0], power_limit)

                            plotgroup_i += 1
                            if ch_idx == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots that...
                                #... can be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                # Shows the figure
                plt.show()

                # Saves the figure
                helpers.save_fig(fig, wind_title, filetype='png', foldername=foldername)



def psd_bandwise(psd, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, plot_layout=[2,3],
                 keys_to_plot=['avg', 'max'], same_y=True, avg_as_equal=True):
    """ Plots frequency band-wise PSDs of the data.

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
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_std : bool, default True
    -   Whether or not to plot standard deviation values (if they are present) alongside the data.

    plot_layout : list of ints of size [1 x 2]
    -   A list specifying how plots should be layed out on the page. plot_layout[0] is the number of rows, and
        plot_layout[1] is the number of columns. [2, 3] by default, i.e. 2 rows and 3 columns.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.

    same_y : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True (default), the same axes are
        used; if False, the same axes are not used.

    avg_as_equal : bool, default True
    -   Whether or not to treat averaged data as equivalent, regardless of what was averaged over. E.g. if some data had
        been averaged across subjects 1 & 2, but other data across subjects 3 & 4, avg_as_equal = True would treat this
        data as if it was the same group of data.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Checks to make sure that S.D. data is present if it is requested
    avg_std_present = False
    max_std_present = False
    if plot_std is True:
        for col in psd.columns:
            if 'avg_std' in col:
                avg_std_present = True
            if 'max_std' in col:
                max_std_present = True
        if avg_std_present == False and max_std_present == False:
            print(f"Warning: Standard deviation data is not present, so it cannot be plotted.")
            plot_std = False

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(psd.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)

    # Sets averaged data to the same type, if requested
    if avg_as_equal == True:
        for key in psd.keys():
            for data_i, data in enumerate(psd[key]):
                if type(data) == str or type(data) == np.str_:
                    if data[:3] == 'avg':
                        psd[key].iloc[data_i] = 'avg'

    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Keys containing data that do not represent different conditions
    psd_data_keys = ['ch_coords', 'ch_coords_std', 'freqs', 'psd', 'psd_std', 'fbands', 'fbands_avg', 'fbands_avg_std',
                     'fbands_max', 'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    data_keys = [key for key in psd.keys() if key not in group_fig+group_plot+psd_data_keys]
    plot_keys = [key for key in psd.keys() if key not in group_fig+psd_data_keys+data_keys]
    fig_keys = [key for key in psd.keys() if key not in psd_data_keys+data_keys+plot_keys]

    ## Alters level of keys depending on the values in the data
    # Moves eligible labels from data to subplot level
    move_up = []
    for key in data_keys:
        if len(np.unique(psd[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    data_keys = [key for key in data_keys if key not in move_up]
    plot_keys += move_up
    plot_keys = pd.unique(plot_keys).tolist()
    # Moves eligible labels from subplot to figure level
    move_up = []
    for key in plot_keys:
        if len(np.unique(psd[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    plot_keys = [key for key in plot_keys if key not in move_up]
    fig_keys += move_up
    fig_keys = pd.unique(fig_keys).tolist()

    # If each plot should contain data from multiple subgroups, make sure these subgroups are binary (e.g. MedOff vs....
    #... MedOn) and only one such group is present (e.g. only med, not med and stim)
    if len(data_keys) > 0:
        if len(data_keys) > 1:
            raise ValueError(f"Data from too many conditions {data_keys} are being plotted on the same plot. Only one is allowed.")
        if len(np.unique(psd[data_keys[0]])) > 2:
            raise ValueError(f"Data of different types ({np.unique(psd[data_keys][0])}) from the {data_keys[0]} condition are being plotted on the same plot, but this is only allowed for binary condition data.")

    # Gets indices of master-grouped data
    names_master = helpers.combine_names(psd, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)

    # Generates key names for the band-wise data based on the requested features in keys_to_plot. Follows the form...
    #... fbands_feature (e.g. fbands_avg)
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')

    # Name of the folder in which to save figures (based on the current time)
    foldername = 'psd_bandwise-'+''.join([str(x) for x in datetime.datetime.now().timetuple()[:-3]])


    ## Plotting
    for mastergroup_i, idc_group_master in enumerate(idcs_group_master):

        # Gets a global y-axis for all data of the same type (if requested)
        if same_y == True:
            if plot_std == True:
                if 'psd_std' in psd.keys():
                    ylim = []
                    for key in fullkeys_to_plot:
                        ylim.extend(helpers.same_axes(psd[key].iloc[idc_group_master] +
                                                      psd[key+'_std'].iloc[idc_group_master]))
                        ylim.extend(helpers.same_axes(psd[key].iloc[idc_group_master] -
                                                      psd[key+'_std'].iloc[idc_group_master]))
                    group_ylim = [min(ylim), max(ylim)]
                else:
                    raise ValueError("Plotting S.D. is requested, but the values are not present in the data.")
            else:
                group_ylim = helpers.same_axes(psd[[key for key in fullkeys_to_plot]].iloc[idc_group_master])

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
            n_plots_per_page = int(plot_layout[0]*plot_layout[1]) # number of plots on each page
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = plot_layout[0] # number of rows these pages will need
            n_cols = plot_layout[1] # number of columns these pages will need

            stop = False
            plotgroup_i = 0
            for page_i in range(n_pages):

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_rows == 1 and n_cols == 1:
                    axs = np.asarray([[axs]]) # puts the lone axs into an array for later indexing
                elif n_rows == 1 and n_cols > 1:
                    axs = np.vstack((axs, [0,0])) # adds an extra row for later indexing
                elif n_cols == 1 and n_rows > 1:
                    axs = np.hstack((axs), [0,0]) # adds and extra column for later indexing
                plt.tight_layout(rect=[0, 0, 1, .95])
                fig.suptitle(wind_title)

                for row_i in range(n_rows): # fill up each row from top to down...
                    for col_i in range(n_cols): # ... and from left to right
                        if stop is False: # if there is still data to plot for this method
                            
                            idc_group_plot = idcs_group_plot[plotgroup_i] # indices of the data entries of this subgroup

                            # Gets the characteristics for all data of this plot so that a title can be generated
                            plot_info = {}
                            for key in plot_keys:
                                plot_info[key] = list(np.unique(psd[key][idc_group_plot]))
                            plot_title, _ = helpers.plot_title(plot_info, already_included=included, full_info=False)

                            # Sets up subplot
                            axs[row_i, col_i].set_title(plot_title)
                            axs[row_i, col_i].set_ylabel('Normalised Power (% total)')

                            if len(data_keys) == 0: # if data of multiple conditions is not being plotted
                            
                                data = psd.iloc[idc_group_plot[0]] # the data to plot

                                # Sets up data for plotting as bars
                                n_groups = len(keys_to_plot)
                                bands = data.fbands
                                n_bars = len(bands)
                                width = 1/n_bars

                                # Location of bars in the groups
                                start_locs = np.arange(n_groups, step=width*(n_bars+2)) # makes sure the bars of each...
                                #... group don't overlap
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
                                            fmaxs.append(str(int(data.fbands_fmax[band_i])))
                                            if max_std_present == True:
                                                fmax_std = u'\u00B1'+str(int(np.ceil(data.fbands_fmax_std[band_i])))
                                                fmaxs[-1] += fmax_std

                                    # Plots the data
                                    alpha = .8
                                    axs[row_i, col_i].bar(bar_locs[band_i], to_plot, width=width, label=band,
                                                          alpha=alpha)
                                    if plot_std == True:
                                        axs[row_i, col_i].errorbar(bar_locs[band_i], to_plot, yerr=stds, capsize=3,
                                                                   fmt=' ', color='black', alpha=alpha)

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(group_ylim[0], group_ylim[1])

                                # Tidies up the x-axis ticks and labels
                                axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                                axs[row_i, col_i].set_xticklabels(keys_to_plot)
                                axs[row_i, col_i].legend(loc='upper left', labelspacing=0)

                                # Adds the fmax data to the bars (if applicable)
                                if 'max' in keys_to_plot:
                                    ylim = axs[row_i, col_i].get_ylim()
                                    for fmax_i, fmax in enumerate(fmaxs):
                                        text_ypos = data.fbands_max[fmax_i]+ylim[1]*.02 # where to put text
                                        if max_std_present == True:
                                            text_ypos += data.fbands_max_std[fmax_i]
                                        # adds the fmax values at an angle to the bars one at a time
                                        axs[row_i, col_i].text(group_locs[1][fmax_i], text_ypos, fmax+'Hz', ha='center',
                                                               rotation=90)
                                    if plot_std == False:
                                        added_height = .15
                                    else:
                                        added_height = .2
                                    axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*added_height]) # increases the subplot height...
                                    #... to accomodate the text

                                plotgroup_i+= 1 # moves on to the next data to plot
                                if idc_group_plot[0] == idc_group_fig[-1]: # if there is no more data to plot for...
                                #... this type...
                                    stop = True #... don't plot anything else
                                    extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots...
                                    #... than can be removed
                            
                            elif len(data_keys) == 1: # if data of multiple conditions is being plotted
                                
                                if len(idc_group_plot) > 2:
                                    raise ValueError(f"Subgroups of data from the condition {data_keys[0]} is being plotted on the same figure. A maximum of data from two subgroups (e.g. MedOff vs. MedOn) is supported, but data from {len(idc_group_plot)} subgroups is being plotted.")

                                data = psd.iloc[idc_group_plot] # the data to plot
                                subgroup_names= [data_keys[0]+x for x in np.unique(psd[data_keys[0]])] # names of the...
                                #... subgroups being plotted (e.g. medOff, medOn)

                                # Sets up data for plotting as bars
                                n_groups = len(keys_to_plot)
                                bands = data.iloc[0].fbands
                                n_bars = len(bands)*len(idc_group_plot) # one bar for each subgroup of each freq band
                                width = 1/n_bars

                                # Location of bars in the groups
                                start_locs = np.arange(n_groups, step=width*(n_bars+2)) # makes sure the bars of each...
                                #... group don't overlap
                                group_locs = []
                                for start_loc in start_locs: # x-axis bar positions, grouped by group
                                    group_locs.append([start_loc+width*i for i in np.arange(n_bars)])
                                bar_locs = []
                                for bar_i in range(n_bars): # x-axis bar positions, grouped by band
                                    bar_locs.append([])
                                    for group_i in range(n_groups):
                                        bar_locs[bar_i].append(group_locs[group_i][bar_i])

                                # Colours of the bars
                                colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(bands)]
                                alphas = [.8, .4]

                                ## Gets the data to plot and plots the data
                                if 'max' in keys_to_plot:
                                    fmaxs = []
                                    for i in idc_group_plot:
                                        fmaxs.append([])

                                data_i = 0
                                for band_i, band in enumerate(bands): # for each frequency band
                                    for ch_i, ch_idx in enumerate(idc_group_plot): # for each piece of data in the group

                                        to_plot = []
                                        if plot_std == True:
                                            stds = []

                                        for key in fullkeys_to_plot:
                                                to_plot.append(data.loc[ch_idx][key][band_i]) # gets the data to be...
                                                #... plotted...
                                                if plot_std == True: #... and the std of this data (if applicable)
                                                    if f'{key}_std' in data.keys():
                                                        stds.append(data.loc[ch_idx][f'{key}_std'][band_i])
                                                    else:
                                                        stds.append(np.nan)

                                                if 'fbands_max' in key: # gets the std of the fmax data to add to the...
                                                #... plots
                                                    fmaxs[ch_i].append(str(int(data.loc[ch_idx].fbands_fmax[band_i])))
                                                    if max_std_present == True:
                                                        fmax_std = u'\u00B1'+str(int(
                                                                    np.ceil(data.loc[ch_idx].fbands_fmax_std[band_i])))
                                                        fmaxs[ch_i][-1] += fmax_std

                                        # Plots the data
                                        if ch_i == 0:
                                            axs[row_i, col_i].bar(bar_locs[data_i], to_plot, width=width, label=band,
                                                                  color=colours[band_i], alpha=alphas[ch_i])
                                        else:
                                            axs[row_i, col_i].bar(bar_locs[data_i], to_plot, width=width,
                                                                  color=colours[band_i], alpha=alphas[ch_i])
                                        if plot_std == True:
                                            axs[row_i, col_i].errorbar(bar_locs[data_i], to_plot, yerr=stds, capsize=3,
                                                                       fmt=' ', color='black', alpha=alphas[ch_i])
                                        data_i += 1

                                # Adds surrogate data for the legend
                                ylim = axs[row_i, col_i].get_ylim()
                                xlim = axs[row_i, col_i].get_xlim()
                                for subgroup_i in range(len(subgroup_names)):
                                    axs[row_i, col_i].scatter(0, -99, label=subgroup_names[subgroup_i],
                                                              color='black', alpha=alphas[subgroup_i])
                                axs[row_i, col_i].set_ylim(ylim)
                                axs[row_i, col_i].set_xlim(xlim)

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(group_ylim[0], group_ylim[1])

                                # Tidies up the x-axis ticks and labels
                                axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                                axs[row_i, col_i].set_xticklabels(keys_to_plot)
                                axs[row_i, col_i].legend(loc='upper left', labelspacing=0)

                                # Adds the fmax data to the bars (if applicable)
                                if 'max' in keys_to_plot:
                                    ylim = axs[row_i, col_i].get_ylim()

                                    sorted_fmaxs = [] # combines fmax values from the two groups (e.g. MedOff vs....
                                    #... MedOn) for alternative plotting (e.g. MedOff value, MedOn value, etc...)
                                    sorted_ypos = [] # generates the y-axis text coordinates using data from the two...
                                    #... groups (e.g. MedOff vs. MedOn) for alternative plotting (e.g. MedOff value,...
                                    #... MedOn Value, etc...)
                                    for fmax_i in range(len(fmaxs[0])):
                                        for ch_i in range(len(fmaxs)):
                                            sorted_fmaxs.append(fmaxs[ch_i][fmax_i])
                                            sorted_ypos.append(data.iloc[ch_i].fbands_max[fmax_i]+ylim[1]*.02)
                                            if max_std_present == True:
                                                sorted_ypos[-1] += data.iloc[ch_i].fbands_max_std[fmax_i]

                                    data_i = 0
                                    for subgroup_i in range(len(fmaxs)):
                                        for fmax_i in range(len(fmaxs[subgroup_i])):
                                            fmax = sorted_fmaxs[data_i]
                                            text_ypos = sorted_ypos[data_i]
                                            # adds the fmax values at an angle to the bars one at a time
                                            axs[row_i, col_i].text(group_locs[1][data_i], text_ypos, fmax+'Hz',
                                                                   ha='center', rotation=90)
                                            data_i += 1
                                    if plot_std == False:
                                        added_height = .15
                                    else:
                                        added_height = .2
                                    axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*added_height]) # increases...
                                    #... the subplot height to accomodate the text

                                plotgroup_i+= 1 # moves on to the next data to plot
                                if idc_group_plot[-1] == idc_group_fig[-1]: # if there is no more data to plot for...
                                #... this type...
                                    stop = True #... don't plot anything else
                                    extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots...
                                    #... than can be removed

                            else:
                                raise ValueError(f"Multiple types of data from different conditions ({data_keys}) are being plotted on the same plot, but this is not allowed.")

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                # Shows the figure
                plt.show()

                # Saves the figure
                helpers.save_fig(fig, wind_title, filetype='png', foldername=foldername)



def psd_bandwise_gb(psd, areas, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_layout=[2,3],
                    keys_to_plot=['avg', 'max'], same_y_groupwise=False, same_y_bandwise=False, normalise=[False, []],
                    avg_as_equal=True):
    """ Plots frequency band-wise PSDs of the data on a glass brain.

    PARAMETERS
    ----------
    psds : pandas DataFrame
    -   A DataFrame containing the channel names, their types, and the normalised power values.

    areas : list of strs
    -   A list of strings containing the area of the brain that is to be plotted (e.g. 'cortical', 'deep'). N.B. Only
        'cortical' is currently supported!!!

    group_master : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate data into groups that share the
        same y-axis limits (if a same_y or normalisation option is requested).

    group_fig : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate the grouped data into
        subgroups used for plotting the same figure(s) with the same title. If empty, the same groups as group_master
        are used.

    group_plot : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate the subgrouped data (specified
        by group_figure) into further subgroups used for plotting data on the same plot. Should only have one entry and 
        be for a binary data characteristic (e.g. 'med': Off & On), otherwise an error is raised. If the data is binary,
        the subtypes are plotted on different hemispheres (e.g. MedOff on left hemisphere, MedOn on right hemisphere).

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_layout : list of ints of size [1 x 2]
    -   A list specifying how plots should be layed out on the page. plot_layout[0] is the number of rows, and
        plot_layout[1] is the number of columns. [2, 3] by default, i.e. 2 rows and 3 columns.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.

    same_y_groupwise : bool, default False
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True, the same axes are
        used; if False (default), the same axes are not used. Cannot be used alongside same_y_bandwise or normalise.

    same_y_bandwise : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type and frequency band. If True
        (default), the same axes are used; if False, the same axes are not used. Cannot be used alongside
        same_y_groupwise or normalise.

    normalise : list of size (1x2)
    -   Information on whether (and if so, how) to normalise data. normalise[0] is a bool specifying whether
        normalisation should occur (True) or not (False). normalise [1] is a list of strings containing data
        characteristics used in addition to the group_master characteristics to group data for
        normalisation (useful for better comparison across e.g. subjects, conditions). If an empty list is given
        (default), the group_master characteristics are used.
    
    avg_as_equal : bool, default True
    -   Whether or not to treat averaged data as equivalent, regardless of what was averaged over. E.g. if some data had
        been averaged across subjects 1 & 2, but other data across subjects 3 & 4, avg_as_equal = True would treat this
        data as if it was the same group of data.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Checks that only one same_y is used
    if same_y_groupwise == True and same_y_bandwise == True:
        raise ValueError("The same y-axes can only be used across groups (same_y_groupwise), or groups and frequency bands (same_y_bandwise), but both have been requested. Set only one to be True.")
    
    # Checks that normalise is given in the correct format
    if not isinstance(normalise[0], bool) or not isinstance(normalise[1], list):
        raise ValueError("The first element of normalise must be a bool, and the second element must be a list of strings.")

    # Checks for correct group_plot inputs and makes adjustments to coordinates (if necessary)
    subgroup_names = []
    if group_plot != []:

        # Checks that only one datatype is provided in group_plot
        if len(group_plot) > 1:
            raise ValueError(f"Only one type of data can be plotted on the same glass brain, but {group_plot} are requested.")

        subgroups = np.unique(psd[group_plot[0]])
        subgroup_names = [group_plot[0]+subgroup for subgroup in subgroups]
        # Checks that there are multiple subgroups (e.g. MedOff & MedOn) to plot
        if len(subgroups) > 1:
            # Checks that the datatype provided in group_plot is binary
            if len(subgroups) > 2:
                raise ValueError(f"The {group_plot[0]} group to plot on the same glass brain is not binary.")
            # Switches the coordinates so that each subgroup (e.g. MedOff vs. MedOn) is plotted on a different...
            #... hemisphere
            for data_i, subgroup in enumerate(psd[group_plot[0]]):
                if psd.ch_coords[data_i][0] > 0 and subgroup == subgroups[0]: # if the x-coord is in the right...
                #... hemisphere and is for data from e.g. group MedOff
                    psd.ch_coords[data_i][0] = psd.ch_coords[data_i][0]*-1 # switch the x-coord to the left hemisphere
                if psd.ch_coords[data_i][0] < 0 and subgroup == subgroups[1]: # if the x-coord is in the left...
                #... hemisphere and is for data from e.g. group MedOn
                    psd.ch_coords[data_i][0] = psd.ch_coords[data_i][0]*-1 # switch the x-coord to the right hemisphere


    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(psd.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)

    # Sets averaged data to the same type, if requested
    if avg_as_equal == True:
        for key in psd.keys():
            for data_i, data in enumerate(psd[key]):
                if type(data) == str or type(data) == np.str_:
                    if data[:3] == 'avg':
                        psd[key].iloc[data_i] = 'avg'

    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Keys containing data that do not represent different conditions
    psd_data_keys = ['ch_coords', 'ch_coords_std', 'freqs', 'psd', 'psd_std', 'fbands', 'fbands_avg', 'fbands_avg_std',
                     'fbands_max', 'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in psd.keys() if key not in psd_data_keys]

    # Keys of the values to plot
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')

    # Name of the folder in which to save figures (based on the current time)
    foldername = 'psd_bandwise_gb-'+''.join([str(x) for x in datetime.datetime.now().timetuple()[:-3]])

    og_psd = psd.copy()
    for area in areas:

        if area == 'cortical':
            vertices = io.loadmat('coherence\\Vertices.mat') # outline of the glass brain
            gb_x = vertices['Vertices'][::1,0] # x-coordinates of the top-down view
            gb_y = vertices['Vertices'][::1,1] # y-coordinates of the top-down view
            coords_key = 'ch_coords' # name of the coordinates in the data
        else:
            raise ValueError("Only 'cortical' is supported as an area to plot as a glass brain currently.")

        # Resets the data to the unaltered version (restores data that may have been removed due it not being in the...
        #... previously processed area)
        psd = og_psd.copy()

        # Discards data not in the requested area
        remove = []
        for i, ch_type in enumerate(psd.ch_type):
            if ch_type != area:
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)

        # Gets indices of master-grouped data
        names_master = helpers.combine_names(psd, group_master, joining=',')
        names_group_master, idcs_group_master = helpers.unique_names(names_master)


        ## Plotting
        for mastergroup_i, idc_group_master in enumerate(idcs_group_master):

            first = True
            for idx in idc_group_master:
                if first == True:
                    fbands = psd.iloc[idc_group_master].fbands[idx]
                    first = False
                else:
                    if fbands != psd.iloc[idc_group_master].fbands[idx] != True:
                        raise ValueError("The frequency bands do not match for data of the same group.")

            # Gets indices of figure-grouped data
            names_fig = helpers.combine_names(psd.iloc[idc_group_master], group_fig, joining=',')
            names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]

            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):

                names_plot = helpers.combine_names(psd, group_plot, joining=',')
                names_group_plot, idcs_group_plot = helpers.unique_names([names_plot[i] for i in idc_group_fig])

                for plot_key in fullkeys_to_plot:

                    ## Normalises the data (if requested) and calculates y-axis limits
                    if normalise[0] == True: # gets the indices of the data to normalise
                        names_norm = helpers.combine_names(psd.iloc[idc_group_master], normalise[1])
                        names_group_norm, idcs_group_norm = helpers.unique_names(names_norm)
                        for normgroup_i, idc_group_norm in enumerate(idcs_group_norm):
                            idcs_group_norm[normgroup_i] = [idc_group_master[i] for i in idc_group_norm]

                        if same_y_groupwise == True:
                            for idc_group_norm in idcs_group_norm:
                                norm_vals = stats.zscore([item for sublist in 
                                                          psd[plot_key].iloc[idc_group_norm].values.flatten()
                                                          for item in sublist]) # gets the normalised values
                                norm_vals = np.reshape(norm_vals, (len(psd[plot_key].iloc[idc_group_norm]),
                                                int(len(norm_vals)/len(psd[plot_key].iloc[idc_group_norm]))))
                                for idx, idx_group_norm in enumerate(idc_group_norm): # assigns the normalised values
                                    psd.iloc[idx_group_norm] = norm_vals[idx]

                        elif same_y_bandwise == True:
                            for idc_group_norm in idcs_group_norm:
                                for fband_i in range(len(fbands)): # gets the normalised values
                                    norm_vals = stats.zscore([list(item)[fband_i] for item in 
                                                              psd[plot_key].iloc[idc_group_norm].values.flatten()])
                                    for idx, idx_group_norm in enumerate(idc_group_norm): # assigns the normalised...
                                    #... values
                                        psd[plot_key].iloc[idx_group_norm][fband_i] = norm_vals[idx]

                    # Gets a global y-axis for all data of the same type (if requested)
                    if same_y_groupwise == True:
                        ylim = helpers.same_axes(psd[plot_key].iloc[idc_group_master]) # gets the y-axis limits

                    # Gets a global y-axis for all data of the same type and frequency band (if requested)
                    if same_y_bandwise == True:
                        ylims = [] # gets the y-axis limits
                        for fband_i in range(len(fbands)):
                            ylims.append(helpers.same_axes([x[fband_i] for x in psd[plot_key].iloc[idc_group_master]]))

                    data = psd.iloc[idc_group_fig]

                    # Gets the characteristics for all data of this type so that a title for the window can be generated
                    fig_info = {}
                    for key in fig_keys:
                        fig_info[key] = list(np.unique(data[key]))
                    wind_title, included = helpers.window_title(fig_info, base_title=f'PSD-{plot_key}:',
                                                                full_info=False)

                    n_plots = len(idcs_group_plot) # number of plots to make for this type
                    n_plots_per_page = int(plot_layout[0]*plot_layout[1]) # number of plots on each page
                    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
                    n_rows = plot_layout[0] # number of rows these pages will need
                    n_cols = plot_layout[1] # number of columns these pages will need

                    stop = False
                    fband_i = 0
                    for page_i in range(n_pages):

                        # Sets up figure
                        fig, axs = plt.subplots(n_rows, n_cols)
                        if n_rows == 1 and n_cols == 1:
                            axs = np.asarray([[axs]]) # puts the lone axs into an array for later indexing
                        elif n_rows == 1 and n_cols > 1:
                            axs = np.vstack((axs, [0,0])) # adds an extra row for later indexing
                        elif n_cols == 1 and n_rows > 1:
                            axs = np.hstack((axs), [0,0]) # adds and extra column for later indexing
                        plt.tight_layout(rect=[0, 0, 1, .95])
                        fig.suptitle(wind_title)

                        for row_i in range(n_rows): # fill up each row from top to down...
                            for col_i in range(n_cols): # ... and from left to right
                                if stop is False: # if there is still data to plot for this method

                                    # Sets up subplot
                                    axs[row_i, col_i].set_title(fbands[fband_i])
                                    axs[row_i, col_i].set_axis_off()

                                    # Adds the glass brain
                                    axs[row_i, col_i].scatter(gb_x, gb_y, c='gray', s=.001)

                                    # Gets the colour bar limits for the particular frequency band
                                    if same_y_bandwise == True:
                                        ylim = ylims[fband_i]

                                    # Plots data on the brain
                                    for idc_group_plot in idcs_group_plot:
                                        if same_y_groupwise == True or same_y_bandwise == True: # sets the colour bar...
                                        #... limits
                                            plotted_data = axs[row_i, col_i].scatter(
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][0] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # x-coords
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][1] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # y-coords
                                                c=[data.iloc[idc_group_plot][plot_key].iloc[i][fband_i] for i in
                                                   range(np.shape(data.iloc[idc_group_plot][plot_key])[0])], # values
                                                s=30, alpha=.8, edgecolor='black', cmap='viridis',
                                                vmin=ylim[0], vmax=ylim[1]
                                            )
                                        else:
                                            plotted_data = axs[row_i, col_i].scatter(
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][0] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # x-coords
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][1] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # y-coords
                                                c=[data.iloc[idc_group_plot][plot_key].iloc[i][fband_i] for i in
                                                   range(np.shape(data.iloc[idc_group_plot][plot_key])[0])], # values
                                                s=30, alpha=.8, edgecolor='black', cmap='viridis'
                                            )
                                    
                                    # Stops brains from getting squashed due to aspect ratio changes
                                    axs[row_i, col_i].set_aspect('equal')

                                    # Adds a colour map to the plot
                                    cbar = fig.colorbar(plotted_data, ax=axs[row_i, col_i])
                                    if normalise[0] == False:
                                        title = 'Normalised Power (% total)'
                                    else:
                                        title = f'Power (z-scored{normalise[1]})'
                                    cbar.set_label(title)
                                    cbar.ax.tick_params(axis='y')

                                    # Adds the name of the subgroup to each hemisphere, if necessary
                                    if len(subgroup_names) > 1:
                                        bot = axs[row_i, col_i].get_ylim()[0]
                                        axs[row_i, col_i].text(.05, bot, f"{subgroup_names[0]} / {subgroup_names[1]}",
                                                               ha='center')

                                    fband_i+= 1 # moves on to the next data to plot
                                    if fband_i == len(fbands): # if there is no more data to plot for this type...
                                        stop = True #... don't plot anything else
                                        extra = n_plots_per_page*n_pages - n_plots # checks if there are extra...
                                        #... subplots that can be removed

                                elif stop is True and extra > 0: # if there is no more data to plot for this type...
                                    fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                        # Shows the figure
                        plt.show()

                        # Saves the figure
                        helpers.save_fig(fig, wind_title, filetype='png', foldername=foldername)



def coh_freqwise(coh, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, plot_layout=[2,3],
                 freq_limit=None, same_y=True, avg_as_equal=True, mark_y0=True):
    """ Plots single-frequency-wise coherence data.

    PARAMETERS
    ----------
    coh : pandas DataFrame
    -   A DataFrame containing the corresponding ECoG and LFP channel names, and the single-frequency-wise coherence
        data.

    group_master : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate data into groups that share the
        same y-axis limits (if applicable)

    group_fig : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate the grouped data into
        subgroups used for plotting the same figure(s) with the same title. If empty, the same groups as group_master
        are used.

    group_plot : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate the subgrouped data (specified
        by group_figure) into further subgroups used for plotting data on the same plot. If empty, the same groups as
        group_fig are used.

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_std : bool, default True
    -   Whether or not to plot standard deviation values (if they are present) alongside the data.

    plot_layout : list of ints of size [1 x 2]
    -   A list specifying how plots should be layed out on the page. plot_layout[0] is the number of rows, and
        plot_layout[1] is the number of columns. [2, 3] by default, i.e. 2 rows and 3 columns.

    freq_limit : int | float
    -   The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.
    
    same_y : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True (default), the same axes are
        used; if False, the same axes are not used.

    avg_as_equal : bool, default True
    -   Whether or not to treat averaged data as equivalent, regardless of what was averaged over. E.g. if some data had
        been averaged across subjects 1 & 2, but other data across subjects 3 & 4, avg_as_equal = True would treat this
        data as if it was the same group of data.

    mark_y0 : bool, default False
    -   Whether or not to draw a dotted line where y = 0. If False (default), no line is drawn. If True, a line is
        drawn.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Checks to make sure that S.D. data is present if it is requested
    if plot_std is True:
        std_present = False
        for col in coh.columns:
            if 'std' in col:
                std_present = True
        if std_present == False:
            print(f"Warning: Standard deviation data is not present, so it cannot be plotted.")
            plot_std = False

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)

    # Sets averaged data to the same type, if requested
    if avg_as_equal == True:
        for key in coh.keys():
            for data_i, data in enumerate(coh[key]):
                if type(data) == str or type(data) == np.str_:
                    if data[:3] == 'avg':
                        coh[key].iloc[data_i] = 'avg'

    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Keys used to label figures, plots, and data
    coh_data_keys = ['ch_coords_cortical', 'ch_coords_deep', 'ch_coords_cortical_std', 'ch_coords_deep_std', 'freqs',
                     'coh', 'coh_std', 'fbands', 'fbands_avg', 'fbands_avg_std', 'fbands_max', 'fbands_max_std',
                     'fbands_fmax', 'fbands_fmax_std']
    data_keys = [key for key in coh.keys() if key not in group_fig+group_plot+coh_data_keys]
    plot_keys = [key for key in coh.keys() if key not in group_fig+coh_data_keys+data_keys]
    fig_keys = [key for key in coh.keys() if key not in coh_data_keys+data_keys+plot_keys]

    ## Alters level of keys depending on the values in the data
    # Moves eligible labels from data to subplot level
    move_up = []
    for key in data_keys:
        if len(np.unique(coh[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    data_keys = [key for key in data_keys if key not in move_up]
    plot_keys += move_up
    plot_keys = pd.unique(plot_keys).tolist()
    # Moves eligible labels from subplot to figure level
    move_up = []
    for key in plot_keys:
        if len(np.unique(coh[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    plot_keys = [key for key in plot_keys if key not in move_up]
    fig_keys += move_up
    fig_keys = pd.unique(fig_keys).tolist()
    
    # Gets indices of master-grouped data
    names_master = helpers.combine_names(coh, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)

    # Gets colours of data
    colour_info = helpers.get_colour_info(coh, [key for key in coh.keys() if key not in coh_data_keys
                                                and key not in group_master and key not in group_fig
                                                and key not in group_plot])
    colours = helpers.data_colour(colour_info, not_for_unique=True, avg_as_equal=avg_as_equal)

    # Name of the folder in which to save figures (based on the current time)
    foldername = 'coh_freqwise-'+''.join([str(x) for x in datetime.datetime.now().timetuple()[:-3]])


    ### Plotting
    for mastergroup_i, idc_group_master in enumerate(idcs_group_master):

        # Gets a global y-axis for all data of the same type (if requested)
        if same_y == True:
            if plot_std == True:
                if 'coh_std' in coh.keys():
                    ylim = []
                    ylim.extend(helpers.same_axes(coh.coh[idc_group_master] + coh.coh_std[idc_group_master]))
                    ylim.extend(helpers.same_axes(coh.coh[idc_group_master] - coh.coh_std[idc_group_master]))
                    group_ylim = [min(ylim), max(ylim)]
                else:
                    raise ValueError("Plotting S.D. is requested, but the values are not present in the data.")
            else:
                group_ylim = helpers.same_axes(coh.coh[idc_group_master])

        # Gets indices of figure-grouped data
        names_fig = helpers.combine_names(coh.iloc[idc_group_master], group_fig, joining=',')
        names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
        for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]

        for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):

            # Gets the characteristics for all data of this type so that a title for the window can be generated
            fig_info = {}
            for key in fig_keys:
                fig_info[key] = list(np.unique(coh[key][idc_group_fig]))
            wind_title, included = helpers.window_title(fig_info, base_title='Coh:', full_info=False)

            names_plot = helpers.combine_names(coh, group_plot, joining=',')
            names_group_plot, idcs_group_plot = helpers.unique_names([names_plot[i] for i in idc_group_fig])
            for plotgroup_i, idc_group_plot in enumerate(idcs_group_plot):
                idcs_group_plot[plotgroup_i] = [idc_group_fig[i] for i in idc_group_plot]

            n_plots = len(idcs_group_plot) # number of plots to make for this type
            n_plots_per_page = int(plot_layout[0]*plot_layout[1]) # number of plots on each page
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = plot_layout[0] # number of rows these pages will need
            n_cols = plot_layout[1] # number of columns these pages will need

            stop = False
            plotgroup_i = 0
            for page_i in range(n_pages): # for each page of this type

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_rows == 1 and n_cols == 1:
                    axs = np.asarray([[axs]]) # puts the lone axs into an array for later indexing
                elif n_rows == 1 and n_cols > 1:
                    axs = np.vstack((axs, [0,0])) # adds an extra row for later indexing
                elif n_cols == 1 and n_rows > 1:
                    axs = np.hstack((axs), [0,0]) # adds and extra column for later indexing
                plt.tight_layout(rect = [0, 0, 1, .95])
                fig.suptitle(wind_title)

                for row_i in range(n_rows): # fill up each row from top to down...
                    for col_i in range(n_cols): # ... and from left to right
                        if stop is False: # if there is still data to plot for this type

                            idc_group_plot = idcs_group_plot[plotgroup_i]

                            # Gets the characteristics for all data of this plot so that a title can be generated
                            plot_info = {}
                            for key in plot_keys:
                                plot_info[key] = list(np.unique(coh[key][idc_group_plot]))
                            plot_title, plot_included = helpers.plot_title(plot_info, already_included=included,
                                                                            full_info=False)
                            now_included = included + plot_included # keeps track of what is already included in the...
                            #... titles

                            # Sets up subplot
                            axs[row_i, col_i].set_title(plot_title)
                            axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                            axs[row_i, col_i].set_ylabel('Coherence')

                            for ch_idx in idc_group_plot: # for each data entry
                            
                                data = coh.iloc[ch_idx] # the data to plot

                                if freq_limit != None: # finds limit of frequencies to plot (if applicable)...
                                    freq_limit_i = int(np.where(data.freqs == data.freqs[data.freqs >=
                                                    freq_limit].min())[0])
                                else: #... or plots all data
                                    freq_limit_i = len(data.freqs)

                                # Gets the characteristics for the data so that a label can be generated
                                data_info = {}
                                for key in data_keys:
                                    data_info[key] = data[key]
                                data_title = helpers.data_title(data_info, already_included=now_included,
                                                                full_info=False)
                                if data_title == '': # don't label the data if there is no info to add
                                    data_title = None

                                # Gets the colour of data based on it's characteristics
                                colour = []
                                for key in colours.keys():
                                    colour.append(colours[key][colour_info[key][1].index(data[key])])
                                if colour:
                                    colour = np.nanmean(colour, axis=0) # takes the average colour based on the...
                                    #... data's characteristics
                                else: # if there is no colour info, set the colour to black
                                    colour = [0, 0, 0, 1]
                                
                                # Plots data
                                axs[row_i, col_i].plot(data.freqs[:freq_limit_i+1], data.coh[:freq_limit_i+1],
                                                       label=data_title, linewidth=2, color=colour)
                                if data_title != None: # if data has been labelled, plot the legend
                                    axs[row_i, col_i].legend(labelspacing=0)
                                
                                # Plots std (if applicable)
                                if plot_std == True:
                                    std_plus = data.coh[:freq_limit_i+1] + data.coh_std[:freq_limit_i+1]
                                    std_minus = data.coh[:freq_limit_i+1] - data.coh_std[:freq_limit_i+1]
                                    axs[row_i, col_i].fill_between(data.freqs[:freq_limit_i+1], std_plus, std_minus,
                                                                   color=colour, alpha=colour[-1]*.2)
                                
                                # Demarcates 0 on the y-axis, if requested
                                if mark_y0 == True:
                                    xlim = axs[row_i, col_i].get_xlim()
                                    xlim_trim = (xlim[1] - xlim[0])*.05
                                    axs[row_i, col_i].plot([xlim[0]+xlim_trim, xlim[1]-xlim_trim], [0,0], color='grey',
                                        linestyle='dashed')

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(*group_ylim)

                            plotgroup_i += 1 # moves on to the next data to plot
                            if ch_idx == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than...
                                #... can be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                # Shows the figure
                plt.show()

                # Saves the figure
                helpers.save_fig(fig, wind_title, filetype='png', foldername=foldername)



def coh_bandwise(coh, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, plot_layout=[2,3],
                 keys_to_plot=['avg', 'max'], same_y=True, avg_as_equal=True):
    """ Plots frequency band-wise coherence data.

    PARAMETERS
    ----------
    coh : pandas DataFrame
    -   A DataFrame containing the corresponding ECoG and LFP channel names, and the frequency band-wise coherence
        data.

    group_master : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate data into groups that share the
        same y-axis limits (if applicable)

    group_fig : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate the grouped data into
        subgroups used for plotting the same figure(s) with the same title. If empty, the same groups as group_master
        are used.

    group_plot : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate the subgrouped data (specified
        by group_figure) into further subgroups used for plotting data on the same plot. If empty, the same groups as
        group_fig are used.

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_std : bool, default True
    -   Whether or not to plot standard deviation values (if they are present) alongside the data.

    plot_layout : list of ints of size [1 x 2]
    -   A list specifying how plots should be layed out on the page. plot_layout[0] is the number of rows, and
        plot_layout[1] is the number of columns. [2, 3] by default, i.e. 2 rows and 3 columns.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.

    same_y : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True (default), the same axes are
        used; if False, the same axes are not used.

    avg_as_equal : bool, default True
    -   Whether or not to treat averaged data as equivalent, regardless of what was averaged over. E.g. if some data had
        been averaged across subjects 1 & 2, but other data across subjects 3 & 4, avg_as_equal = True would treat this
        data as if it was the same group of data.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Checks to make sure that S.D. data is present if it is requested
    if plot_std is True:
        for col in coh.columns:
            if 'avg_std' in col:
                avg_std_present = True
            if 'max_std' in col:
                max_std_present = True
        if avg_std_present == False and max_std_present == False:
            print(f"Warning: Standard deviation data is not present, so it cannot be plotted.")
            plot_std = False
    avg_std_present = False
    max_std_present = False

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)

    # Sets averaged data to the same type, if requested
    if avg_as_equal == True:
        for key in coh.keys():
            for data_i, data in enumerate(coh[key]):
                if type(data) == str or type(data) == np.str_:
                    if data[:3] == 'avg':
                        coh[key].iloc[data_i] = 'avg'

    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Keys used to label figures, plots, and data
    coh_data_keys = ['ch_coords_cortical', 'ch_coords_deep', 'ch_coords_cortical_std', 'ch_coords_deep_std', 'freqs',
                     'coh', 'coh_std', 'fbands', 'fbands_avg', 'fbands_avg_std', 'fbands_max', 'fbands_max_std',
                     'fbands_fmax', 'fbands_fmax_std']
    data_keys = [key for key in coh.keys() if key not in group_fig+group_plot+coh_data_keys]
    plot_keys = [key for key in coh.keys() if key not in group_fig+coh_data_keys+data_keys]
    fig_keys = [key for key in coh.keys() if key not in coh_data_keys+data_keys+plot_keys]

    ## Alters level of keys depending on the values in the data
    # Moves eligible labels from data to subplot level
    move_up = []
    for key in data_keys:
        if len(np.unique(coh[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    data_keys = [key for key in data_keys if key not in move_up]
    plot_keys += move_up
    plot_keys = pd.unique(plot_keys).tolist()
    # Moves eligible labels from subplot to figure level
    move_up = []
    for key in plot_keys:
        if len(np.unique(coh[key])) == 1: # if there is only one type of data present...
            move_up.append(key) #... move this data to a higher labelling level (avoids label redundancy and clutter)
    plot_keys = [key for key in plot_keys if key not in move_up]
    fig_keys += move_up
    fig_keys = pd.unique(fig_keys).tolist()

    # If each plot should contain data from multiple subgroups, make sure these subgroups are binary (e.g. MedOff vs....
    #... MedOn) and only one such group is present (e.g. only med, not med and stim)
    if len(data_keys) > 0:
        if len(data_keys) > 1:
            raise ValueError(f"Data from too many conditions {data_keys} are being plotted on the same plot. Only one is allowed.")
        if len(np.unique(coh[data_keys[0]])) > 2:
            raise ValueError(f"Data of different types ({np.unique(coh[data_keys][0])}) from the {data_keys[0]} condition are being plotted on the same plot, but this is only allowed for binary condition data.")


    # Gets indices of master-grouped data
    names_master = helpers.combine_names(coh, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)

    # Generates key names for the band-wise data based on the requested features in keys_to_plot. Follows the form...
    #... fbands_feature (e.g. fbands_avg)
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')

    # Name of the folder in which to save figures (based on the current time)
    foldername = 'coh_bandwise-'+''.join([str(x) for x in datetime.datetime.now().timetuple()[:-3]])


    ## Plotting
    for mastergroup_i, idc_group_master in enumerate(idcs_group_master):

        # Gets a global y-axis for all data of the same type (if requested)
        if same_y == True:
            if plot_std == True:
                if 'coh_std' in coh.keys():
                    ylim = []
                    for key in fullkeys_to_plot:
                        ylim.extend(helpers.same_axes(coh[key].iloc[idc_group_master] +
                                                      coh[key+'_std'].iloc[idc_group_master]))
                        ylim.extend(helpers.same_axes(coh[key].iloc[idc_group_master] -
                                                      coh[key+'_std'].iloc[idc_group_master]))
                    group_ylim = [min(ylim), max(ylim)]
                else:
                    raise ValueError("Plotting S.D. is requested, but the values are not present in the data.")
            else:
                group_ylim = helpers.same_axes(coh[[key for key in fullkeys_to_plot]].iloc[idc_group_master])

        # Gets indices of figure-grouped data
        names_fig = helpers.combine_names(coh.iloc[idc_group_master], group_fig, joining=',')
        names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
        for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]

        for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):

            # Gets the characteristics for all data of this type so that a title for the window can be generated
            fig_info = {}
            for key in fig_keys:
                fig_info[key] = list(np.unique(coh[key][idc_group_fig]))
            wind_title, included = helpers.window_title(fig_info, base_title='Coh:', full_info=False)

            names_plot = helpers.combine_names(coh, group_plot, joining=',')
            names_group_plot, idcs_group_plot = helpers.unique_names([names_plot[i] for i in idc_group_fig])
            for plotgroup_i, idc_group_plot in enumerate(idcs_group_plot):
                idcs_group_plot[plotgroup_i] = [idc_group_fig[i] for i in idc_group_plot]

            n_plots = len(idcs_group_plot) # number of plots to make for this type
            n_plots_per_page = int(plot_layout[0]*plot_layout[1]) # number of plots on each page
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = plot_layout[0] # number of rows these pages will need
            n_cols = plot_layout[1] # number of columns these pages will need

            stop = False
            plotgroup_i = 0
            for page_i in range(n_pages):

                # Sets up figure
                fig, axs = plt.subplots(n_rows, n_cols)
                if n_rows == 1 and n_cols == 1:
                    axs = np.asarray([[axs]]) # puts the lone axs into an array for later indexing
                elif n_rows == 1 and n_cols > 1:
                    axs = np.vstack((axs, [0,0])) # adds an extra row for later indexing
                elif n_cols == 1 and n_rows > 1:
                    axs = np.hstack((axs), [0,0]) # adds and extra column for later indexing
                plt.tight_layout(rect=[0, 0, 1, .95])
                fig.suptitle(wind_title)

                for row_i in range(n_rows): # fill up each row from top to down...
                    for col_i in range(n_cols): # ... and from left to right
                        if stop is False: # if there is still data to plot for this method
                            
                            idc_group_plot = idcs_group_plot[plotgroup_i] # indices of the data entries of this subgroup

                            # Gets the characteristics for all data of this plot so that a title can be generated
                            plot_info = {}
                            for key in plot_keys:
                                plot_info[key] = list(np.unique(coh[key][idc_group_plot]))
                            plot_title, _ = helpers.plot_title(plot_info, already_included=included, full_info=False)

                            # Sets up subplot
                            axs[row_i, col_i].set_title(plot_title)
                            axs[row_i, col_i].set_ylabel('Coherence')

                            if len(data_keys) == 0: # if data of multiple conditions is not being plotted
                            
                                data = coh.iloc[idc_group_plot[0]] # the data to plot

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
                                            fmaxs.append(str(int(data.fbands_fmax[band_i])))
                                            if max_std_present == True:
                                                fmax_std = u'\u00B1'+str(int(np.ceil(data.fbands_fmax_std[band_i])))
                                                fmaxs[-1] += fmax_std

                                    # Plots the data
                                    alpha=.8
                                    axs[row_i, col_i].bar(bar_locs[band_i], to_plot, width=width, label=band,
                                                         alpha=alpha)
                                    if plot_std == True:
                                        axs[row_i, col_i].errorbar(bar_locs[band_i], to_plot, yerr=stds, capsize=3,
                                                                   fmt=' ', color='black', alpha=alpha)

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(group_ylim[0], group_ylim[1])

                                # Tidies up the x-axis ticks and labels
                                axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                                axs[row_i, col_i].set_xticklabels(keys_to_plot)
                                axs[row_i, col_i].legend(loc='upper left', labelspacing=0)

                                # Adds the fmax data to the bars (if applicable)
                                if 'max' in keys_to_plot:
                                    ylim = axs[row_i, col_i].get_ylim()
                                    for fmax_i, fmax in enumerate(fmaxs):
                                        text_ypos = data.fbands_max[fmax_i]+ylim[1]*.01 # where to put text
                                        if max_std_present == True:
                                            text_ypos += data.fbands_max_std[fmax_i]
                                        # adds the fmax values at an angle to the bars one at a time
                                        axs[row_i, col_i].text(group_locs[1][fmax_i], text_ypos, fmax+'Hz', ha='center',
                                                               rotation=60)
                                    axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*.07]) # increases the subplot height...
                                    #... to accomodate the text

                                plotgroup_i+= 1 # moves on to the next data to plot
                                if idc_group_plot[0] == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                    stop = True #... don't plot anything else
                                    extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                    #... be removed

                            elif len(data_keys) == 1:

                                if len(idc_group_plot) > 2:
                                    raise ValueError(f"Subgroups of data from the condition {data_keys[0]} is being plotted on the same figure. A maximum of data from two subgroups (e.g. MedOff vs. MedOn) is supported, but data from {len(idc_group_plot)} subgroups is being plotted.")

                                data = coh.iloc[idc_group_plot] # the data to plot
                                subgroup_names= [data_keys[0]+x for x in np.unique(coh[data_keys[0]])] # names of the...
                                #... subgroups being plotted (e.g. medOff, medOn)

                                # Sets up data for plotting as bars
                                n_groups = len(keys_to_plot)
                                bands = data.iloc[0].fbands
                                n_bars = len(bands)*len(idc_group_plot) # one bar for each subgroup of each freq band
                                width = 1/n_bars

                                # Location of bars in the groups
                                start_locs = np.arange(n_groups, step=width*(n_bars+2)) # makes sure the bars of each...
                                #... group don't overlap
                                group_locs = []
                                for start_loc in start_locs: # x-axis bar positions, grouped by group
                                    group_locs.append([start_loc+width*i for i in np.arange(n_bars)])
                                bar_locs = []
                                for bar_i in range(n_bars): # x-axis bar positions, grouped by band
                                    bar_locs.append([])
                                    for group_i in range(n_groups):
                                        bar_locs[bar_i].append(group_locs[group_i][bar_i])

                                # Colours of the bars
                                colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(bands)]
                                alphas = [.8, .4]

                                ## Gets the data to plot and plots the data
                                if 'max' in keys_to_plot:
                                    fmaxs = []
                                    for i in idc_group_plot:
                                        fmaxs.append([])

                                data_i = 0
                                for band_i, band in enumerate(bands): # for each frequency band
                                    for ch_i, ch_idx in enumerate(idc_group_plot): # for each piece of data in the group

                                        to_plot = []
                                        if plot_std == True:
                                            stds = []

                                        for key in fullkeys_to_plot:
                                                to_plot.append(data.loc[ch_idx][key][band_i]) # gets the data to be...
                                                #... plotted...
                                                if plot_std == True: #... and the std of this data (if applicable)
                                                    if f'{key}_std' in data.keys():
                                                        stds.append(data.loc[ch_idx][f'{key}_std'][band_i])
                                                    else:
                                                        stds.append(np.nan)

                                                if 'fbands_max' in key: # gets the std of the fmax data to add to the...
                                                #... plots
                                                    fmaxs[ch_i].append(str(int(data.loc[ch_idx].fbands_fmax[band_i])))
                                                    if max_std_present == True:
                                                        fmax_std = u'\u00B1'+str(int(
                                                                   np.ceil(data.loc[ch_idx].fbands_fmax_std[band_i])))
                                                        fmaxs[ch_i][-1] += fmax_std

                                        # Plots the data
                                        if ch_i == 0:
                                            axs[row_i, col_i].bar(bar_locs[data_i], to_plot, width=width, label=band,
                                                                  color=colours[band_i], alpha=alphas[ch_i])
                                        else:
                                            axs[row_i, col_i].bar(bar_locs[data_i], to_plot, width=width,
                                                                  color=colours[band_i], alpha=alphas[ch_i])
                                        if plot_std == True:
                                            axs[row_i, col_i].errorbar(bar_locs[data_i], to_plot, yerr=stds, capsize=3,
                                                                       fmt=' ', color='black', alpha=alphas[ch_i])
                                        data_i += 1

                                # Adds surrogate data for the legend
                                ylim = axs[row_i, col_i].get_ylim()
                                xlim = axs[row_i, col_i].get_xlim()
                                for subgroup_i in range(len(subgroup_names)):
                                    axs[row_i, col_i].scatter(0, -99, label=subgroup_names[subgroup_i],
                                                              color='black', alpha=alphas[subgroup_i])
                                axs[row_i, col_i].set_ylim(ylim)
                                axs[row_i, col_i].set_xlim(xlim)

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(group_ylim[0], group_ylim[1])

                                # Tidies up the x-axis ticks and labels
                                axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                                axs[row_i, col_i].set_xticklabels(keys_to_plot)
                                axs[row_i, col_i].legend(loc='upper left', labelspacing=0)

                                # Adds the fmax data to the bars (if applicable)
                                if 'max' in keys_to_plot:
                                    ylim = axs[row_i, col_i].get_ylim()

                                    sorted_fmaxs = [] # combines fmax values from the two groups (e.g. MedOff vs....
                                    #... MedOn) for alternative plotting (e.g. MedOff value, MedOn value, etc...)
                                    sorted_ypos = [] # generates the y-axis text coordinates using data from the two...
                                    #... groups (e.g. MedOff vs. MedOn) for alternative plotting (e.g. MedOff value,...
                                    #... MedOn Value, etc...)
                                    for fmax_i in range(len(fmaxs[0])):
                                        for ch_i in range(len(fmaxs)):
                                            sorted_fmaxs.append(fmaxs[ch_i][fmax_i])
                                            sorted_ypos.append(data.iloc[ch_i].fbands_max[fmax_i]+ylim[1]*.02)
                                            if max_std_present == True:
                                                sorted_ypos[-1] += data.iloc[ch_i].fbands_max_std[fmax_i]

                                    data_i = 0
                                    for subgroup_i in range(len(fmaxs)):
                                        for fmax_i in range(len(fmaxs[subgroup_i])):
                                            fmax = sorted_fmaxs[data_i]
                                            text_ypos = sorted_ypos[data_i]
                                            # adds the fmax values at an angle to the bars one at a time
                                            axs[row_i, col_i].text(group_locs[1][data_i], text_ypos, fmax+'Hz',
                                                                   ha='center', rotation=90)
                                            data_i += 1
                                    if plot_std == False:
                                        added_height = .1
                                    else:
                                        added_height = .2
                                    axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*added_height]) # increases...
                                    #... the subplot height to accomodate the text

                                plotgroup_i+= 1 # moves on to the next data to plot
                                if idc_group_plot[-1] == idc_group_fig[-1]: # if there is no more data to plot for...
                                #... this type...
                                    stop = True #... don't plot anything else
                                    extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots...
                                    #... than can be removed

                            else:
                                raise ValueError(f"Multiple types of data from different conditions ({data_keys}) are being plotted on the same plot, but this is not allowed.")


                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                # Shows the figure
                plt.show()

                # Saves the figure
                helpers.save_fig(fig, wind_title, filetype='png', foldername=foldername)



def coh_bandwise_gb(coh, areas, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_layout=[2,3],
                    keys_to_plot=['avg', 'max'], same_y_groupwise=False, same_y_bandwise=True, normalise=[False,[]],
                    avg_as_equal=True):
    """ Plots frequency band-wise coherence of the data on a glass brain.

    PARAMETERS
    ----------
    coh : pandas DataFrame
    -   A DataFrame containing the corresponding ECoG and LFP channel names, and the frequency band-wise coherence
        data.

    areas : list of strs
    -   A list of strings containing the area of the brain that is to be plotted (e.g. 'cortical', 'deep'). N.B. Only
        'cortical' is currently supported!!!

    group_master : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate data into groups that share the
        same y-axis limits (if applicable)

    group_fig : list of strs
    -   Keys of coh containing the data characteristics which should be used to separate the grouped data into
        subgroups used for plotting the same figure(s) with the same title. If empty, the same groups as group_master
        are used.

    group_plot : list of strs
    -   Keys of psd containing the data characteristics which should be used to separate the subgrouped data (specified
        by group_figure) into further subgroups used for plotting data on the same plot. Should only have one entry and 
        be for a binary data characteristic (e.g. 'med': Off & On), otherwise an error is raised. If the data is binary,
        the subtypes are plotted on different hemispheres (e.g. MedOff on left hemisphere, MedOn on right hemisphere).

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    plot_layout : list of ints of size [1 x 2]
    -   A list specifying how plots should be layed out on the page. plot_layout[0] is the number of rows, and
        plot_layout[1] is the number of columns. [2, 3] by default, i.e. 2 rows and 3 columns.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.

    same_y_groupwise : bool, default False
    -   Whether or not to use the same y-axis boundaries for data of the same type. If True, the same axes are
        used; if False (default), the same axes are not used.

    same_y_bandwise : bool, default True
    -   Whether or not to use the same y-axis boundaries for data of the same type and frequency band. If True
        (default), the same axes are used; if False, the same axes are not used.

    normalise : list of size (1x2)
    -   Information on whether (and if so, how) to normalise data. normalise[0] is a bool specifying whether
        normalisation should occur (True) or not (False). normalise [1] is a list of strings containing data
        characteristics used in addition to the group_master characteristics to group data for
        normalisation (useful for better comparison across e.g. subjects, conditions). If an empty list is given
        (default), the group_master characteristics are used.

    avg_as_equal : bool, default True
    -   Whether or not to treat averaged data as equivalent, regardless of what was averaged over. E.g. if some data had
        been averaged across subjects 1 & 2, but other data across subjects 3 & 4, avg_as_equal = True would treat this
        data as if it was the same group of data.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Checks that only one same_y is used
    if same_y_groupwise == True and same_y_bandwise == True:
        raise ValueError("The same y-axes can only be used across groups (same_y_groupwise), or groups and frequency bands (same_y_bandwise), but both have been requested. Set only one to be True.")

    # Checks for correct group_plot inputs and makes adjustments to coordinates (if necessary)
    subgroup_names = []
    if group_plot != []:

        # Checks that only one datatype is provided in group_plot
        if len(group_plot) > 1:
            raise ValueError(f"Only one type of data can be plotted on the same glass brain, but {group_plot} are requested.")

        subgroups = np.unique(coh[group_plot[0]])
        subgroup_names = [group_plot[0]+subgroup for subgroup in subgroups]
        # Checks that there are multiple subgroups (e.g. MedOff & MedOn) to plot
        if len(subgroups) > 1:
            # Checks that the datatype provided in group_plot is binary
            if len(subgroups) > 2:
                raise ValueError(f"The {group_plot[0]} group to plot on the same glass brain is not binary.")
            # Switches the coordinates so that each subgroup (e.g. MedOff vs. MedOn) is plotted on a different hemisphere
            coords_keys = []
            for area in areas:
                coords_keys.append(f"ch_coords_{area}")
            for coords_key in coords_keys:
                for data_i, subgroup in enumerate(coh[group_plot[0]]):
                    if coh[coords_key][data_i][0] > 0 and subgroup == subgroups[0]: # if the x-coord is in the right...
                    #... hemisphere and is for data from e.g. group MedOff
                        coh[coords_key][data_i][0] = coh[coords_key][data_i][0]*-1 # switch the x-coord to the left...
                    #... hemisphere
                    if coh[coords_key][data_i][0] < 0 and subgroup == subgroups[1]: # if the x-coord is in the left...
                    #... hemisphere and is for data from e.g. group MedOn
                        coh[coords_key][data_i][0] = coh[coords_key][data_i][0]*-1 # switch the x-coord to the right...
                    #... hemisphere

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)

    # Sets averaged data to the same type, if requested
    if avg_as_equal == True:
        for key in coh.keys():
            for data_i, data in enumerate(coh[key]):
                if type(data) == str or type(data) == np.str_:
                    if data[:3] == 'avg':
                        coh[key].iloc[data_i] = 'avg'

    # Establishes groups
    if group_fig == []:
        group_fig = group_master
    if group_plot == []:
        group_plot = group_fig

    # Keys containing data that do not represent different conditions
    coh_data_keys = ['ch_coords_cortical', 'ch_coords_deep', 'ch_coords_cortical_std', 'ch_coords_deep_std', 'freqs',
                     'coh', 'coh_std', 'fbands', 'fbands_avg', 'fbands_avg_std', 'fbands_max', 'fbands_max_std',
                     'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in coh.keys() if key not in coh_data_keys]

    # Keys of the values to plot
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')
    
    # Name of the folder in which to save figures (based on the current time)
    foldername = 'coh_bandwise_gb-'+''.join([str(x) for x in datetime.datetime.now().timetuple()[:-3]])

    for area in areas:

        if area == 'cortical':
            vertices = io.loadmat('coherence\\Vertices.mat') # outline of the glass brain
            gb_x = vertices['Vertices'][::1,0] # x-coordinates of the top-down view
            gb_y = vertices['Vertices'][::1,1] # y-coordinates of the top-down view
            coords_key = f'ch_coords_{area}' # name of the coordinates in the data
        else:
            raise ValueError("Only 'cortical' is supported as an area to plot as a glass brain currently.")

        # Gets indices of master-grouped data
        names_master = helpers.combine_names(coh, group_master, joining=',')
        names_group_master, idcs_group_master = helpers.unique_names(names_master)


        ## Plotting
        for mastergroup_i, idc_group_master in enumerate(idcs_group_master):

            first = True
            for idx in idc_group_master:
                if first == True:
                    fbands = coh.iloc[idc_group_master].fbands[idx]
                    first = False
                else:
                    if fbands != coh.iloc[idc_group_master].fbands[idx] != True:
                        raise ValueError("The frequency bands do not match for data of the same group.")

            # Gets indices of figure-grouped data
            names_fig = helpers.combine_names(coh.iloc[idc_group_master], group_fig, joining=',')
            names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                    idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]

            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):

                names_plot = helpers.combine_names(coh, group_plot, joining=',')
                names_group_plot, idcs_group_plot = helpers.unique_names([names_plot[i] for i in idc_group_fig])

                for plot_key in fullkeys_to_plot:

                    ## Normalises the data (if requested) and calculates y-axis limits
                    if normalise[0] == True: # gets the indices of the data to normalise
                        names_norm = helpers.combine_names(coh.iloc[idc_group_master], normalise[1])
                        names_group_norm, idcs_group_norm = helpers.unique_names(names_norm)
                        for normgroup_i, idc_group_norm in enumerate(idcs_group_norm):
                            idcs_group_norm[normgroup_i] = [idc_group_master[i] for i in idc_group_norm]
                            
                        if same_y_groupwise == True:
                            for idc_group_norm in idcs_group_norm:
                                norm_vals = stats.zscore([item for sublist in 
                                                          coh[plot_key].iloc[idc_group_norm].values.flatten()
                                                          for item in sublist]) # gets the normalised values
                                norm_vals = np.reshape(norm_vals, (len(coh[plot_key].iloc[idc_group_norm]),
                                                int(len(norm_vals)/len(coh[plot_key].iloc[idc_group_norm]))))
                                for idx, idx_group_norm in enumerate(idc_group_norm): # assigns the normalised values
                                    coh.iloc[idx_group_norm] = norm_vals[idx]

                        elif same_y_bandwise == True:
                            for idc_group_norm in idcs_group_norm:
                                for fband_i in range(len(fbands)): # gets the normalised values
                                    norm_vals = stats.zscore([list(item)[fband_i] for item in 
                                                              coh[plot_key].iloc[idc_group_norm].values.flatten()])
                                    for idx, idx_group_norm in enumerate(idc_group_norm): # assigns the normalised...
                                    #... values
                                        coh[plot_key].iloc[idx_group_norm][fband_i] = norm_vals[idx]

                    # Gets a global y-axis for all data of the same type (if requested)
                    if same_y_groupwise == True:
                        ylim = helpers.same_axes(coh[plot_key].iloc[idc_group_master]) # gets the y-axis limits

                    # Gets a global y-axis for all data of the same type and frequency band (if requested)
                    if same_y_bandwise == True:
                        ylims = [] # gets the y-axis limits
                        for fband_i in range(len(fbands)):
                            ylims.append(helpers.same_axes([x[fband_i] for x in coh[plot_key].iloc[idc_group_master]]))

                    data = coh.iloc[idc_group_fig]

                    # Gets the characteristics for all data of this type so that a title for the window can be generated
                    fig_info = {}
                    for key in fig_keys:
                        fig_info[key] = list(np.unique(data[key]))
                    wind_title, _ = helpers.window_title(fig_info, base_title=f'Coh-{plot_key}:',
                                                                full_info=False)

                    n_plots = len(idcs_group_plot) # number of plots to make for this type
                    n_plots_per_page = int(plot_layout[0]*plot_layout[1]) # number of plots on each page
                    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
                    n_rows = plot_layout[0] # number of rows these pages will need
                    n_cols = plot_layout[1] # number of columns these pages will need

                    stop = False
                    fband_i = 0
                    for page_i in range(n_pages):

                        # Sets up figure
                        fig, axs = plt.subplots(n_rows, n_cols)
                        if n_rows == 1 and n_cols == 1:
                            axs = np.asarray([[axs]]) # puts the lone axs into an array for later indexing
                        elif n_rows == 1 and n_cols > 1:
                            axs = np.vstack((axs, [0,0])) # adds an extra row for later indexing
                        elif n_cols == 1 and n_rows > 1:
                            axs = np.hstack((axs), [0,0]) # adds and extra column for later indexing
                        plt.tight_layout(rect=[0, 0, 1, .95])
                        fig.suptitle(wind_title)

                        for row_i in range(n_rows): # fill up each row from top to down...
                            for col_i in range(n_cols): # ... and from left to right
                                if stop is False: # if there is still data to plot for this method

                                    # Sets up subplot
                                    axs[row_i, col_i].set_title(fbands[fband_i])
                                    axs[row_i, col_i].set_axis_off()

                                    # Adds the glass brain
                                    axs[row_i, col_i].scatter(gb_x, gb_y, c='gray', s=.001)

                                    # Gets the colour bar limits for the particular frequency band
                                    if same_y_bandwise == True:
                                        ylim = ylims[fband_i]

                                    # Plots data on the brain
                                    for idc_group_plot in idcs_group_plot:
                                        if same_y_groupwise == True or same_y_bandwise == True: # sets the colour bar limits
                                            plotted_data = axs[row_i, col_i].scatter(
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][0] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # x-coords
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][1] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # y-coords
                                                c=[data.iloc[idc_group_plot][plot_key].iloc[i][fband_i] for i in
                                                   range(np.shape(data.iloc[idc_group_plot][plot_key])[0])], # values
                                                s=30, alpha=.8, edgecolor='black', cmap='viridis',
                                                vmin=ylim[0], vmax=ylim[1]
                                            )
                                        else:
                                            plotted_data = axs[row_i, col_i].scatter(
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][0] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # x-coords
                                                [data.iloc[idc_group_plot][coords_key].iloc[i][1] for i in
                                                 range(np.shape(data.iloc[idc_group_plot][coords_key])[0])], # y-coords
                                                c=[data.iloc[idc_group_plot][plot_key].iloc[i][fband_i] for i in
                                                   range(np.shape(data.iloc[idc_group_plot][plot_key])[0])], # values
                                                s=30, alpha=.8, edgecolor='black', cmap='viridis'
                                            )

                                    # Stops brains from getting squashed due to aspect ratio changes
                                    axs[row_i, col_i].set_aspect('equal')

                                    # Adds a colour map to the plot
                                    cbar = fig.colorbar(plotted_data, ax=axs[row_i, col_i])
                                    if normalise[0] == False:
                                        title = 'Coherence'
                                    else:
                                        title = f'Coherence (z-scored{normalise[1]})'
                                    cbar.set_label(title)
                                    cbar.ax.tick_params(axis='y')

                                    # Adds the name of the subgroup to each hemisphere, if necessary
                                    if len(subgroup_names) > 1:
                                        bot = axs[row_i, col_i].get_ylim()[0]
                                        axs[row_i, col_i].text(.05, bot, f"{subgroup_names[0]} / {subgroup_names[1]}",
                                                               ha='center')


                                    fband_i+= 1 # moves on to the next data to plot
                                    if fband_i == len(fbands): # if there is no more data to plot for this type...
                                        stop = True #... don't plot anything else
                                        extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                        #... be removed

                                elif stop is True and extra > 0: # if there is no more data to plot for this type...
                                    fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                        # Shows the figure
                        plt.show()

                        # Saves the figure
                        helpers.save_fig(fig, wind_title, filetype='png', foldername=foldername)