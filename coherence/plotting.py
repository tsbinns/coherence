import matplotlib.pyplot as plt
import numpy as np


def psd(psds, plot_shuffled=False, n_plots_per_page=6, freq_limit=50):
    

    # Discards shuffled data from being plotted, if requested
    if plot_shuffled is False:
        keys = list(psds.keys())
        remove = []
        for i, name in enumerate(psds['ch_name']):
            if name[:8] == 'SHUFFLED':
                remove.append(i)
        for key in keys:
            if isinstance(psds[key], list):
                psds[key].pop(*[i for i in remove])
            elif isinstance(psds[key], np.ndarray):
                psds[key] = np.delete(psds[key], remove, axis=0)
                
    

    # Gets types of channels present in psds and their indexes
    types = set(psds['ch_type'])
    types_idx = []
    for i, type in enumerate(types):
        types_idx.append([])
        for j, type2 in enumerate(psds['ch_type']):
            if type2 is type:
                types_idx[i].append(j)
    
    # Plotting
    psds_i = 0 # index of the data in psds to plot
    for type_i, type in enumerate(types):

        n_plots = len(types_idx[type_i]) # number of plots to make for this type
        n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
        n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
        n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

        stop = False
        for page_i in range(n_pages): # for each page of this type

            fig, axs = plt.subplots(n_rows, n_cols)
            plt.tight_layout(rect = [0, 0, 1, .97])
            fig.suptitle(type + ' channel PSDs')

            for row_i in range(n_rows): # fill up each row from top to down...
                for col_i in range(n_cols): # ... and from left to right
                    if stop is False: # if there is still data to plot for this type
                        
                        freq_limit_i = len(psds['freqs'][psds_i])
                        if freq_limit != None:
                            freq_limit_i = np.argmin(np.abs(psds['freqs'][psds_i]-freq_limit))

                        mean = np.mean(psds['psd'][psds_i][:freq_limit_i],1)
                        std = np.std(psds['psd'][psds_i][:freq_limit_i],1)

                        axs[row_i, col_i].fill_between(psds['freqs'][psds_i][:freq_limit_i], mean+std, mean-std,
                                                       color='orange', alpha=.3)
                        axs[row_i, col_i].plot(psds['freqs'][psds_i][:freq_limit_i], mean, color='orange', linewidth=2)
                        axs[row_i, col_i].set_title(psds['ch_name'][psds_i])
                        axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                        axs[row_i, col_i].set_ylabel('Power')

                        psds_i += 1 # moves on to the next data to plot in psds
                        if psds_i == len(psds['ch_name']): # if there is no more data to plot for this type
                            stop = True
                            extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can be removed

                    elif stop is True and extra > 0: # if there is no more data to plot for this type...
                        fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

            plt.show()


def coherence(cohs, n_plots_per_page=6, freq_limit=50):
    

    # Gets ECoG channels to plot coherence values of and their indexes
    cortical_chs = np.unique(cohs['ch_name_cortical']).tolist()
    cortical_chs_idx = []
    for i, cortical_ch in enumerate(cortical_chs):
        cortical_chs_idx.append([])
        for j, ch in enumerate(cohs['ch_name_cortical']):
            if ch == cortical_ch:
                cortical_chs_idx[i].append(j)


    # Plotting
    colors = ['orange', 'grey']

    n_plots = len(cortical_chs) # number of plots to make for this type
    n_pages = int(np.ceil(n_plots/n_plots_per_page)) # number of pages these plots will need
    n_rows = int(np.sqrt(n_plots_per_page)) # number of rows these pages will need
    n_cols = int(np.ceil(n_plots_per_page/n_rows)) # number of columns these pages will need

    stop = False
    cohs_i = 0 # index of the data in cohs to plot
    cort_i = 0 # index of the cortical channels to plot
    for page_i in range(n_pages): # for each page

        fig, axs = plt.subplots(n_rows, n_cols)
        plt.tight_layout(rect = [0, 0, 1, .97])
        fig.suptitle('Coherence')

        for row_i in range(n_rows): # fill up each row from top to down...
            for col_i in range(n_cols): # ... and from left to right
                if stop is False: # if there is still data to plot
                    for val_i in range(len(cortical_chs_idx[cort_i])): # for each coherence value for the ECoG channel
                        
                        freq_limit_i = len(cohs['freqs'][cohs_i])
                        if freq_limit != None:
                            freq_limit_i = np.argmin(np.abs(cohs['freqs'][cohs_i]-freq_limit))

                        axs[row_i, col_i].plot(cohs['freqs'][cohs_i][:freq_limit_i], cohs['coh'][cohs_i][:freq_limit_i],
                                               color=colors[val_i], linewidth=2, label=cohs['ch_name_deep'][cohs_i])
                        axs[row_i, col_i].legend()

                        cohs_i += 1 # moves on to the next data to plot in cohs
                        if cohs_i == len(cohs['ch_name_cortical']): # if there is no more data to plot
                            stop = True
                            extra = n_plots_per_page*n_pages - n_plots # checks if there are extra subplots than can be removed

                    axs[row_i, col_i].set_title(cortical_chs[cort_i])
                    axs[row_i, col_i].set_xlabel('Frequency (Hz)')
                    axs[row_i, col_i].set_ylabel('Coherence')
                    cort_i += 1 # moves on to the next cortical channel

                elif stop is True and extra > 0: # if there is no more data to plot for this type...
                    fig.delaxes(axs[row_i, col_i]) # ... delete the extra subplots

        plt.show()

