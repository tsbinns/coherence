import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import io
import helpers


def psd_freqwise(psd, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, n_plots_per_page=6,
                 freq_limit=None, power_limit=None, same_y=True):
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

    # Keys containing data that do not represent different conditions
    psd_data_keys = ['ch_coords', 'freqs', 'psd', 'psd_std', 'fbands', 'fbands_avg', 'fbands_avg_std', 'fbands_max',
                     'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
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
                            now_included = included + plot_included # keeps track of what is already included in the titles

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
                                data_title = helpers.data_title(data_info, already_included=now_included, full_info=False)
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
                            if ch_idx == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                #... be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()



def psd_bandwise(psd, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, n_plots_per_page=6,
                 keys_to_plot=['avg', 'max'], same_y=True):
    """ Plots frequency band-wise PSDs of the data.

    PARAMETERS
    ----------
    psds : pandas DataFrame
    -   A DataFrame containing the channel names, their types, and the normalised power values.

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

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.


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

    # Keys containing data that do not represent different conditions
    psd_data_keys = ['ch_coords', 'freqs', 'psd', 'psd_std', 'fbands', 'fbands_avg', 'fbands_avg_std', 'fbands_max',
                     'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in psd.keys() if key not in psd_data_keys and key not in group_plot]
    plot_keys = [key for key in psd.keys() if key not in psd_data_keys and key not in group_fig]

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

    # Generates key names for the band-wise data based on the requested features in keys_to_plot. Follows the form...
    #... fbands_feature (e.g. fbands_avg)
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')


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
            wind_title, included = helpers.window_title(fig_info, base_title='Coh:', full_info=False)

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

                            for ch_idx in idc_group_plot: # for each data entry
                            
                                data = psd.iloc[ch_idx] # the data to plot

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
                                            if plot_std == True:
                                                fmax_std = u'\u00B1'+str(int(np.ceil(data.fbands_fmax_std[band_i])))
                                                fmaxs[-1] += fmax_std

                                    # Plots the data
                                    axs[row_i, col_i].bar(bar_locs[band_i], to_plot, width=width, label=band, alpha=.7)
                                    if plot_std == True:
                                        axs[row_i, col_i].errorbar(bar_locs[band_i], to_plot, yerr=stds, capsize=3,
                                                                   fmt=' ')

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(group_ylim[0], group_ylim[1])

                                # Tidies up the x-axis ticks and labels
                                axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                                axs[row_i, col_i].set_xticklabels(keys_to_plot)
                                axs[row_i, col_i].legend()

                                # Adds the fmax data to the bars (if applicable)
                                if 'max' in keys_to_plot:
                                    ylim = axs[row_i, col_i].get_ylim()
                                    for fmax_i, fmax in enumerate(fmaxs):
                                        text_ypos = data.fbands_max[fmax_i]+ylim[1]*.01 # where to put text
                                        if plot_std == True:
                                            text_ypos += data.fbands_max_std[fmax_i]
                                        # adds the fmax values at an angle to the bars one at a time
                                        axs[row_i, col_i].text(group_locs[1][fmax_i], text_ypos, fmax+'Hz', ha='center',
                                                               rotation=60)
                                    axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*.07]) # increases the subplot height...
                                    #... to accomodate the text

                            plotgroup_i+= 1 # moves on to the next data to plot
                            if ch_idx == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                #... be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()



def psd_bandwise_gb(psd, areas, group_master, group_fig=[], plot_shuffled=False, n_plots_per_page=6,
                    keys_to_plot=['avg', 'max'], same_y=True):
    """ Plots frequency band-wise PSDs of the data on a glass brain.

    PARAMETERS
    ----------
    psds : pandas DataFrame
    -   A DataFrame containing the channel names, their types, and the normalised power values.

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

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Establishes groups
    if group_fig == []:
        group_fig = group_master

    # Keys containing data that do not represent different conditions
    psd_data_keys = ['ch_coords', 'freqs', 'psd', 'psd_std', 'fbands', 'fbands_avg', 'fbands_avg_std', 'fbands_max',
                     'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in psd.keys() if key not in psd_data_keys]

    # Keys of the values to plot
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(psd.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)
    
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

            # Gets indices of figure-grouped data
            names_fig = helpers.combine_names(psd.iloc[idc_group_master], group_fig, joining=',')
            names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                    idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]

            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):

                data = psd.iloc[idc_group_fig]

                first = True
                for idx in idc_group_fig:
                    if first == True:
                        fbands = data.fbands[idx]
                        first = False
                    else:
                        if fbands != data.fbands[idx] != True:
                            raise ValueError("The frequency bands do not match for data of the same group.")

                for plot_key in fullkeys_to_plot:

                    # Gets a global y-axis for all data of the same type (if requested)
                    if same_y == True:
                        group_ylim = helpers.same_axes(psd[plot_key].iloc[idc_group_master])

                    # Gets the characteristics for all data of this type so that a title for the window can be generated
                    fig_info = {}
                    for key in fig_keys:
                        fig_info[key] = list(np.unique(data[key]))
                    wind_title, included = helpers.window_title(fig_info, base_title=f'PSD-{plot_key}:',
                                                                full_info=False)

                    n_plots = len(fbands) # number of plots to make for this type
                    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
                    n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
                    n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

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

                                    # Plots data on the brain
                                    if same_y == True: # sets the colour bar limits
                                        plotted_data = axs[row_i, col_i].scatter(
                                            [data[coords_key].iloc[i][0]*1000 for i in range(np.shape(data[coords_key])[0])], # x-coords
                                            [data[coords_key].iloc[i][1]*1000 for i in range(np.shape(data[coords_key])[0])], # y-coords
                                            c=[data[plot_key].iloc[i][fband_i] for i in range(np.shape(data[plot_key])[0])], # values
                                            s=30, alpha=.8, edgecolor='black', cmap='viridis', vmin=group_ylim[0], vmax=group_ylim[1]
                                        )
                                    else:
                                        plotted_data = axs[row_i, col_i].scatter(
                                            [data[coords_key].iloc[i][0]*1000 for i in range(np.shape(data[coords_key])[0])], # x-coords
                                            [data[coords_key].iloc[i][1]*1000 for i in range(np.shape(data[coords_key])[0])], # y-coords
                                            c=[data[plot_key].iloc[i][fband_i] for i in range(np.shape(data[plot_key])[0])], # values
                                            s=30, alpha=.8, edgecolor='black', cmap='viridis'
                                        )

                                    # Adds a colour map to the plot
                                    cbar = fig.colorbar(plotted_data, ax=axs[row_i, col_i])
                                    cbar.set_label('Normalised Power (% total)')
                                    cbar.ax.tick_params(axis='y')
                                    #cbar.ax.set_yticklabels(labels=np.round(cbar.get_ticks(),2))

                                    fband_i+= 1 # moves on to the next data to plot
                                    if fband_i == len(fbands): # if there is no more data to plot for this type...
                                        stop = True #... don't plot anything else
                                        extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                        #... be removed

                                elif stop is True and extra > 0: # if there is no more data to plot for this type...
                                    fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                        plt.show()



def coh_freqwise(coh, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, n_plots_per_page=6,
                 freq_limit=None, same_y=True):
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

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.

    freq_limit : int | float
    -   The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.


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

    # Keys containing data that do not represent different conditions
    coh_data_keys = ['ch_coords_cortical', 'ch_coords_deep', 'freqs', 'coh', 'coh_std', 'fbands', 'fbands_avg',
                     'fbands_avg_std', 'fbands_max', 'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in coh.keys() if key not in coh_data_keys and key not in group_plot]
    plot_keys = [key for key in coh.keys() if key not in coh_data_keys and key not in group_fig]
    data_keys = [key for key in coh.keys() if key not in coh_data_keys and key not in group_fig]

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)
    
    # Gets indices of master-grouped data
    names_master = helpers.combine_names(coh, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)

    # Gets colours of data
    colour_info = helpers.get_colour_info(coh, [key for key in coh.keys() if key not in coh_data_keys
                                                and key not in group_master and key not in group_fig
                                                and key not in group_plot])
    colours = helpers.data_colour(colour_info, not_for_unique=True, avg_as_equal=True)


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
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
            n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

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
                            now_included = included + plot_included # keeps track of what is already included in the titles

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
                                data_title = helpers.data_title(data_info, already_included=now_included, full_info=False)
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
                                axs[row_i, col_i].plot(data.freqs[:freq_limit_i+1], data.coh[:freq_limit_i+1],
                                                        label=data_title, linewidth=2, color=colour)
                                if data_title != None: # if data has been labelled, plot the legend
                                    axs[row_i, col_i].legend()
                                
                                # Plots std (if applicable)
                                if plot_std == True:
                                    std_plus = data.coh[:freq_limit_i+1] + data.coh_std[:freq_limit_i+1]
                                    std_minus = data.coh[:freq_limit_i+1] - data.coh_std[:freq_limit_i+1]
                                    axs[row_i, col_i].fill_between(data.freqs[:freq_limit_i+1], std_plus, std_minus,
                                                                    color=colour, alpha=colour[-1]*.2)
                                
                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(*group_ylim)

                            plotgroup_i += 1 # moves on to the next data to plot
                            if ch_idx == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                #... be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()



def coh_bandwise(coh, group_master, group_fig=[], group_plot=[], plot_shuffled=False, plot_std=True, n_plots_per_page=6,
                 keys_to_plot=['avg', 'max'], same_y=True):
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

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.


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

    # Keys containing data that do not represent different conditions
    coh_data_keys = ['ch_coords_cortical', 'ch_coords_deep', 'freqs', 'coh', 'coh_std', 'fbands', 'fbands_avg',
                     'fbands_avg_std', 'fbands_max', 'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in coh.keys() if key not in coh_data_keys and key not in group_plot]
    plot_keys = [key for key in coh.keys() if key not in coh_data_keys and key not in group_fig]

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)

    # Gets indices of master-grouped data
    names_master = helpers.combine_names(coh, group_master, joining=',')
    names_group_master, idcs_group_master = helpers.unique_names(names_master)

    # Generates key names for the band-wise data based on the requested features in keys_to_plot. Follows the form...
    #... fbands_feature (e.g. fbands_avg)
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')


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
            n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
            n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
            n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

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

                            for ch_idx in idc_group_plot: # for each data entry
                            
                                data = coh.iloc[ch_idx] # the data to plot

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
                                            if plot_std == True:
                                                fmax_std = u'\u00B1'+str(int(np.ceil(data.fbands_fmax_std[band_i])))
                                                fmaxs[-1] += fmax_std

                                    # Plots the data
                                    axs[row_i, col_i].bar(bar_locs[band_i], to_plot, width=width, label=band, alpha=.7)
                                    if plot_std == True:
                                        axs[row_i, col_i].errorbar(bar_locs[band_i], to_plot, yerr=stds, capsize=3, fmt=' ')

                                # Sets all y-axes to be equal (if requested)
                                if same_y == True:
                                    axs[row_i, col_i].set_ylim(group_ylim[0], group_ylim[1])

                                # Tidies up the x-axis ticks and labels
                                axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                                axs[row_i, col_i].set_xticklabels(keys_to_plot)
                                axs[row_i, col_i].legend()

                                # Adds the fmax data to the bars (if applicable)
                                if 'max' in keys_to_plot:
                                    ylim = axs[row_i, col_i].get_ylim()
                                    for fmax_i, fmax in enumerate(fmaxs):
                                        text_ypos = data.fbands_max[fmax_i]+ylim[1]*.01 # where to put text
                                        if plot_std == True:
                                            text_ypos += data.fbands_max_std[fmax_i]
                                        # adds the fmax values at an angle to the bars one at a time
                                        axs[row_i, col_i].text(group_locs[1][fmax_i], text_ypos, fmax+'Hz', ha='center',
                                                               rotation=60)
                                    axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*.07]) # increases the subplot height...
                                    #... to accomodate the text

                            plotgroup_i+= 1 # moves on to the next data to plot
                            if ch_idx == idc_group_fig[-1]: # if there is no more data to plot for this type...
                                stop = True #... don't plot anything else
                                extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                #... be removed

                        elif stop is True and extra > 0: # if there is no more data to plot for this type...
                            fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                plt.show()



def coh_bandwise_gb(coh, areas, group_master, group_fig=[], plot_shuffled=False, n_plots_per_page=6,
                    keys_to_plot=['avg', 'max'], same_y=True):
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

    plot_shuffled : bool, default False
    -   Whether or not to plot coherence values for the shuffled LFP data.

    n_plots_per_page : int
    -   The number of subplots to include on each page. 6 by default.
        
    keys_to_plot : list of strs
    -   The keys of the band-wise values to plot.


    RETURNS
    ----------
    N/A
    """

    ### Setup
    # Establishes groups
    if group_fig == []:
        group_fig = group_master

    # Keys containing data that do not represent different conditions
    coh_data_keys = ['ch_coords_cortical', 'ch_coords_deep', 'freqs', 'coh', 'coh_std', 'fbands', 'fbands_avg',
                     'fbands_avg_std', 'fbands_max', 'fbands_max_std', 'fbands_fmax', 'fbands_fmax_std']
    fig_keys = [key for key in coh.keys() if key not in coh_data_keys]

    # Keys of the values to plot
    fullkeys_to_plot = []
    for key in keys_to_plot:
        fullkeys_to_plot.append(f'fbands_{key}')

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, data_type in enumerate(coh.data_type):
            if data_type == 'shuffled':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)
    
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

            # Gets indices of figure-grouped data
            names_fig = helpers.combine_names(coh.iloc[idc_group_master], group_fig, joining=',')
            names_group_fig, idcs_group_fig = helpers.unique_names(names_fig)
            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):
                    idcs_group_fig[figgroup_i] = [idc_group_master[i] for i in idc_group_fig]

            for figgroup_i, idc_group_fig in enumerate(idcs_group_fig):

                data = coh.iloc[idc_group_fig]

                first = True
                for idx in idc_group_fig:
                    if first == True:
                        fbands = data.fbands[idx]
                        first = False
                    else:
                        if fbands != data.fbands[idx] != True:
                            raise ValueError("The frequency bands do not match for data of the same group.")

                for plot_key in fullkeys_to_plot:

                    # Gets a global y-axis for all data of the same type (if requested)
                    if same_y == True:
                        group_ylim = helpers.same_axes(coh[plot_key].iloc[idc_group_master])

                    # Gets the characteristics for all data of this type so that a title for the window can be generated
                    fig_info = {}
                    for key in fig_keys:
                        fig_info[key] = list(np.unique(data[key]))
                    wind_title, _ = helpers.window_title(fig_info, base_title=f'Coh-{plot_key}:',
                                                                full_info=False)

                    n_plots = len(fbands) # number of plots to make for this type
                    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
                    n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
                    n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

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

                                    # Plots data on the brain
                                    if same_y == True: # sets the colour bar limits
                                        plotted_data = axs[row_i, col_i].scatter(
                                            [data[coords_key].iloc[i][0]*1000 for i in range(np.shape(data[coords_key])[0])], # x-coords
                                            [data[coords_key].iloc[i][1]*1000 for i in range(np.shape(data[coords_key])[0])], # y-coords
                                            c=[data[plot_key].iloc[i][fband_i] for i in range(np.shape(data[plot_key])[0])], # values
                                            s=30, alpha=.8, edgecolor='black', cmap='viridis', vmin=group_ylim[0], vmax=group_ylim[1]
                                        )
                                    else:
                                        plotted_data = axs[row_i, col_i].scatter(
                                            [data[coords_key].iloc[i][0]*1000 for i in range(np.shape(data[coords_key])[0])], # x-coords
                                            [data[coords_key].iloc[i][1]*1000 for i in range(np.shape(data[coords_key])[0])], # y-coords
                                            c=[data[plot_key].iloc[i][fband_i] for i in range(np.shape(data[plot_key])[0])], # values
                                            s=30, alpha=.8, edgecolor='black', cmap='viridis'
                                        )

                                    # Adds a colour map to the plot
                                    cbar = fig.colorbar(plotted_data, ax=axs[row_i, col_i])
                                    cbar.set_label('Coherence')
                                    cbar.ax.tick_params(axis='y')
                                    #cbar.ax.set_yticklabels(labels=np.round(cbar.get_ticks(),2))

                                    fband_i+= 1 # moves on to the next data to plot
                                    if fband_i == len(fbands): # if there is no more data to plot for this type...
                                        stop = True #... don't plot anything else
                                        extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                                        #... be removed

                                elif stop is True and extra > 0: # if there is no more data to plot for this type...
                                    fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

                        plt.show()