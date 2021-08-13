import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import full
import helpers


def psd(psd, plot_shuffled=False, plot_std=True, n_plots_per_page=6, freq_limit=None):
    """ Plots PSDs of the data.

    PARAMETERS
    ----------
    psds : pandas DataFrame
        A DataFrame containing the channel names, their types, and the normalised power values.
    plot_shuffled : bool, default False
        Whether or not to plot PSDs for the shuffled LFP data.
    n_plots_per_page : int
        The number of subplots to include on each page. 6 by default.
    freq_limit : int | float
        The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.

    RETURNS
    ----------
    N/A
    
    """

    ### Setup
    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, name in enumerate(psd.ch_name):
            if name[:8] == 'SHUFFLED':
                remove.append(i)
        psd.drop(remove, inplace=True)
        psd.reset_index(drop=True, inplace=True)
    
    # Gets unique channel names and their indices
    unique_ch_names, unique_ch_idc = helpers.unique_channel_names(list(psd.ch_name))
    _, unique_types_idc = helpers.unique_channel_types(unique_ch_idc, list(psd.ch_type))
    
    # Gets colours of data
    colour_info = {
        'med': ['binary', list(np.unique(psd.med))],
        'stim': ['binary', list(np.unique(psd.stim))],
        'task': ['non-binary', list(np.unique(psd.task))],
        'subject': ['non-binary', list(np.unique(psd.subject))],
        'run': ['non-binary', list(np.unique(psd.run))]
    }
    colours = helpers.data_colour(colour_info)

    ### Plotting
    unique_ch_i = 0 # keeps track of the channel whose data is being plotted
    for type_idc in unique_types_idc: # for each type

        unique_ch_i_oftype = 0 # keeps track of the channel in each type whose data is being plotted

        n_plots = len(type_idc) # number of plots to make for this type
        n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
        n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
        n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need   

        # Gets the indices of all data of this type
        all_type_idc = []
        for ch_idc in type_idc:
            all_type_idc.extend(ch_idc)

        # Gets the characteristics for all data of this type so that a title for the window can be generated
        type_info = {
            'med': list(np.unique(psd.med[all_type_idc])),
            'stim': list(np.unique(psd.stim[all_type_idc])),
            'task': list(np.unique(psd.task[all_type_idc])),
            'subject': list(np.unique(psd.subject[all_type_idc])),
            'run': list(np.unique(psd.run[all_type_idc]))
        }
        wind_title, included = helpers.window_title(type_info, base_title='PSD:')

        stop = False
        for page_i in range(n_pages): # for each page of this type

            # Sets up figure
            fig, axs = plt.subplots(n_rows, n_cols)
            plt.tight_layout(rect = [0, 0, 1, .97])
            fig.suptitle(wind_title)

            for row_i in range(n_rows): # fill up each row from top to down...
                for col_i in range(n_cols): # ... and from left to right
                    if stop is False: # if there is still data to plot for this type

                        ch_idc = type_idc[unique_ch_i_oftype] # indices of the data entries of this channel

                        # Gets the characteristics for all data of this channel so that a title for the subplot can...
                        #... be generated
                        channel_info = {
                            'med': list(np.unique(psd.med[ch_idc])),
                            'stim': list(np.unique(psd.stim[ch_idc])),
                            'task': list(np.unique(psd.task[ch_idc])),
                            'subject': list(np.unique(psd.subject[ch_idc])),
                            'run': list(np.unique(psd.run[ch_idc]))
                        }
                        ch_title, ch_included = helpers.channel_title(channel_info, already_included=included,
                                                          base_title=unique_ch_names[unique_ch_i]+':')
                        included.extend(ch_included) # keeps track of what is already included in the titles

                        # Sets up subplot
                        axs[row_i, col_i].set_title(ch_title)
                        axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                        axs[row_i, col_i].set_ylabel('Normalised Power (% total)')

                        for ch_idx in ch_idc: # for each data entry
                        
                            data = psd.iloc[ch_idx] # the data to plot

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
                                'run': data.run
                            }
                            data_title = helpers.data_title(data_info, already_included=included)
                            if data_title == '': # don't label the data if there is no info to add
                                data_title = None

                            # Gets the colour of data based on it's characteristics
                            colour = []
                            for key in colour_info.keys():
                                colour.append(colours[key][colour_info[key][1].index(data[key])])
                            colour = np.mean(colour, axis=0)[0] # takes the average colour based on the data's...
                            #... characteristics
                            
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
                                                                color=colour, alpha=.3)

                        unique_ch_i += 1 # moves on to the next data to plot
                        unique_ch_i_oftype += 1 # moves on to the next data to plot
                        if unique_ch_i > len(type_idc): # if there is no more data to plot for this type...
                            stop = True #... don't plot anything else
                            extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                            #... be removed

                    elif stop is True and extra > 0: # if there is no more data to plot for this type...
                        fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

            plt.show()




def coherence_fwise(coh, plot_shuffled=False, plot_std=True, n_plots_per_page=6, freq_limit=None,
                    methods=['coh', 'imcoh']):
    """ Plots single-frequency-wise coherence data.

    PARAMETERS
    ----------
    coh : pandas DataFrame
        A DataFrame containing the corresponding ECoG and LFP channel names, and the single-frequency-wise coherence
        data.
    plot_shuffled : bool, default False
        Whether or not to plot coherence values for the shuffled LFP data.
    plot_std : bool, default True
        Whether or not to plot standard deviation values (if they are present) alongside the data.
    n_plots_per_page : int
        The number of subplots to include on each page. 6 by default.
    freq_limit : int | float
        The frequency (in Hz) at which to stop plotting data. If None (default), up to the maximum frequency in the data
        is plotted.
    methods : list of str
        The methods used to calculate coherence. By default, 'coh' (standard coherence) and 'imcoh' (imaginary
        coherence)

    RETURNS
    ----------
    N/A
    
    """

    ### Setup
    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, name in enumerate(coh.ch_name_deep):
            if name[:8] == 'SHUFFLED':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)
    
    # Gets unique channel names and their indices
    unique_ch_names, unique_ch_idc = helpers.unique_channel_names(list(coh.ch_name_cortical))
    
    # Gets colours of data
    colour_info = {
        'med': ['binary', list(np.unique(coh.med))],
        'stim': ['binary', list(np.unique(coh.stim))],
        'task': ['non-binary', list(np.unique(coh.task))],
        'subject': ['non-binary', list(np.unique(coh.subject))],
        'run': ['non-binary', list(np.unique(coh.run))]
    }
    colours = helpers.data_colour(colour_info)

    ### Plotting
    n_plots = len(unique_ch_names) # number of plots to make for this type
    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
    n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
    n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need   

    # Gets the characteristics for all the data so that a title for the window can be generated
    dataset_info = {
        'med': list(np.unique(coh.med)),
        'stim': list(np.unique(coh.stim)),
        'task': list(np.unique(coh.task)),
        'subject': list(np.unique(coh.subject)),
        'run': list(np.unique(coh.run))
    }

    for method in methods:

        unique_ch_i = 0 # keeps track of the channel whose data is being plotted
        wind_title, wind_included = helpers.window_title(dataset_info, base_title=f'{method}:')
        stop = False

        for page_i in range(n_pages): # for each page of this type

            # Sets up figure
            fig, axs = plt.subplots(n_rows, n_cols)
            plt.tight_layout(rect = [0, 0, 1, .97])
            fig.suptitle(wind_title)

            for row_i in range(n_rows): # fill up each row from top to down...
                for col_i in range(n_cols): # ... and from left to right
                    if stop is False: # if there is still data to plot for this type

                        ch_idc = unique_ch_idc[unique_ch_i] # indices of the data entries of this channel

                        # Gets the characteristics for all data of this channel so that a title for the subplot can...
                        #... be generated
                        channel_info = {
                            'med': list(np.unique(coh.med[ch_idc])),
                            'stim': list(np.unique(coh.stim[ch_idc])),
                            'task': list(np.unique(coh.task[ch_idc])),
                            'subject': list(np.unique(coh.subject[ch_idc])),
                            'run': list(np.unique(coh.run[ch_idc])),
                            ':': list(np.unique(coh.ch_name_deep[ch_idc]))
                        }
                        ch_title, ch_included = helpers.channel_title(channel_info, already_included=wind_included,
                                                            base_title=unique_ch_names[unique_ch_i]+':')
                        included = [*wind_included, *ch_included]

                        # Sets up subplot
                        axs[row_i, col_i].set_title(ch_title)
                        axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                        axs[row_i, col_i].set_ylabel('Coherence')

                        for ch_idx in ch_idc: # for each data entry
                        
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
                                ':': data.ch_name_deep
                            }
                            data_title = helpers.data_title(data_info, already_included=included)
                            if data_title == '': # don't label the data if there is no info to add
                                data_title = None

                            # Gets the colour of data based on it's characteristics
                            colour = []
                            for key in colour_info.keys():
                                colour.append(colours[key][colour_info[key][1].index(data[key])])
                            colour = np.mean(colour, axis=0)[0] # takes the average colour based on the data's...
                            #... characteristics
                            
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

                        unique_ch_i += 1 # moves on to the next data to plot
                        if unique_ch_i > len(unique_ch_idc): # if there is no more data to plot for this type...
                            stop = True #... don't plot anything else
                            extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                            #... be removed

                    elif stop is True and extra > 0: # if there is no more data to plot for this type...
                        fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

            plt.show()


def coherence_bandwise(coh, plot_shuffled=False, plot_std=True, n_plots_per_page=6, methods=['coh', 'imcoh'],
                       keys_to_plot=['avg', 'max']):
    """ Plots frequency band-wise coherence data.

    PARAMETERS
    ----------
    coh : pandas DataFrame
        A DataFrame containing the corresponding ECoG and LFP channel names, and the frequency band-wise coherence
        data.
    plot_shuffled : bool, default False
        Whether or not to plot coherence values for the shuffled LFP data.
    plot_std : bool, default True
        Whether or not to plot standard deviation values (if they are present) alongside the data.
    n_plots_per_page : int
        The number of subplots to include on each page. 6 by default.
    methods : list of strs
        The methods used to calculate coherence. By default, 'coh' (standard coherence) and 'imcoh' (imaginary
        coherence).
    keys_to_plot : list of strs
        The keys of the band-wise values to plot.

    RETURNS
    ----------
    N/A
    
    """

    ### Setup
    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        remove = []
        for i, name in enumerate(coh.ch_name_deep):
            if name[:8] == 'SHUFFLED':
                remove.append(i)
        coh.drop(remove, inplace=True)
        coh.reset_index(drop=True, inplace=True)

    # Gets ECoG-LFP combined channel names
    comb_ch_names = helpers.combine_channel_names(coh, ['ch_name_cortical', 'ch_name_deep'], joining=' - ')

    n_plots = len(coh.ch_name_cortical) # number of plots to make for this type
    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
    n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
    n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need   

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

        ch_i = 0 # index of data to plot
        wind_title, included = helpers.window_title(dataset_info, base_title=f'{method}:') # title of the window
        stop = False

        # Generates key names for the band-wise data based on the requested features in keys_to_plot. Follows the...
        #... form method_fbands_feature (e.g. coh_fbands_avg)
        fullkeys_to_plot = []
        for key in keys_to_plot:
            fullkeys_to_plot.append(f'{method}_fbands_{key}')

        for page_i in range(n_pages):

            # Sets up figure
            fig, axs = plt.subplots(n_rows, n_cols)
            plt.tight_layout(rect=[0, 0, 1, .95])
            fig.suptitle(wind_title)

            for row_i in range(n_rows): # fill up each row from top to down...
                for col_i in range(n_cols): # ... and from left to right
                    if stop is False: # if there is still data to plot for this method

                        data = coh.iloc[ch_i] # the data to plot

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
                                                                      base_title=f'{comb_ch_names[ch_i]}:')

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
                                    if plot_std == True:
                                        fmax_std = u'\u00B1'+str(int(np.ceil(data[method+'_fbands_fmax_std'][band_i])))
                                        fmaxs[-1] += fmax_std

                            # Plots the data
                            axs[row_i, col_i].bar(bar_locs[band_i], to_plot, width=width, label=band, alpha=.7)
                            if plot_std == True:
                                axs[row_i, col_i].errorbar(bar_locs[band_i], to_plot, yerr=stds, capsize=3, fmt=' ')

                        # Tidies up the x-axis ticks and labels
                        axs[row_i, col_i].set_xticks((start_locs-width/2)+(width*(n_bars/2)))
                        axs[row_i, col_i].set_xticklabels(keys_to_plot)
                        axs[row_i, col_i].legend()

                        # Adds the fmax data to the bars (if applicable)
                        if 'max' in keys_to_plot:
                            ylim = axs[row_i, col_i].get_ylim()
                            for fmax_i, fmax in enumerate(fmaxs):
                                axs[row_i, col_i].text(group_locs[1][fmax_i], data[method+'_fbands_max'][fmax_i]+
                                                       data[method+'_fbands_max_std'][fmax_i]+ylim[1]*.01, fmax,
                                                       ha='center', rotation=60) # adds the fmax values at an angle to the bars...
                                #... one at a time
                            axs[row_i, col_i].set_ylim([ylim[0], ylim[1]+ylim[1]*.07]) # increases the subplot height...
                            #... to accomodate the text

                        ch_i += 1 # moves on to the next data to plot
                        if ch_i > np.shape(coh)[0]: # if there is no more data to plot for this type...
                            stop = True #... don't plot anything else
                            extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can...
                            #... be removed

                    elif stop is True and extra > 0: # if there is no more data to plot for this type...
                        fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

            plt.show()

