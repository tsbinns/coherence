import numpy as np
import pandas as pd
import os
from copy import deepcopy
from matplotlib import pyplot as plt
from fooof import FOOOF
from scipy import stats as st



def filter_for_annotation(data, highpass=3, lowpass=125, line_noise=50):
    """ Basic filtering of data that is to be annotated. Useful for seeing whether artefacts in the data will disappear
    with filtering.

    PARAMETERS
    ----------
    data : MNE Raw object
    -   The data that is to be filtered.

    highpass : int | float
    -   The highpass frequency (in Hz) to use in the bandpass filter. Default 3 Hz.

    lowpass : int | float
    -   The lowpass frequency (in Hz) to use in the bandpass filter. Default 125 Hz.

    line_noise : int | float
    -   The line noise frequency (in Hz) of the recording, used to generate a notch filter for the line noise and its
        harmonics. Default 50 Hz.


    RETURNS
    ----------
    data : MNE Raw object
    -   The filtered data.
    """

    data.load_data()

    # Notch filters data
    notch = np.arange(line_noise, lowpass, line_noise)
    data.notch_filter(notch, n_jobs='cuda')

    # Bandpass filters data
    data.filter(highpass, lowpass)


    return data



def combine_data(data, info):
    """ Combines data from different factors (e.g. runs, subjects, tasks) together into a single pandas DataFrame.

    PARAMETERS
    ----------
    data : list of pandas DataFrames
    -   The data that is to be combined. The data should correspond to the factor being combined over.
    -   E.g. If you want to combine the data together from multiple runs, the data in each DataFrame should consist of
        the data from a single run.
    -   E.g. If you want to combine the data together from multiple subjects, the data in each DataFrame
        should consist of the data from a single subject.

    info : dict with one key and one value (list of strs)
    -   The factor to be combined over. The key (should only be one) corresponds to the factor, and the value should be
        a string of the IDs of this factor, with a length equal to that of the data list.
    -   E.g. info = {'runs': ['01', '02']} would mean data[0] would have a column named 'runs' with entries '01' added,
        and data[1] would have a column named 'runs' with entried '02' added.


    RETURNS
    ----------
    all_data : pandas DataFrame
    -   The data that has been combined.


    NOTES
    ----------
    -   Can also be used to add tags to single sets of data (i.e. not combining any data, but still adding descriptive
        tags) if only a single set of data is given in a list.
    """

    ### Sanity checks
    # Checks that data is in a list
    if type(data) != list:
        raise ValueError('The data to combine must be in a list.')
    # Checks for only one input
    if len(info.keys()) > 1:
        raise ValueError('Data can only be combined over one factor at a time (e.g. runs, subjects, tasks, etc...).')


    ### Combines data
    # Adds IDs to data of same type (e.g. data from same runs)
    key = list(info.keys())[0]
    for i, x in enumerate(info[key]):
        data_id = list(np.repeat(x, np.shape(data[i])[0])) # makes ID tags of appropriate length
        data_id = pd.DataFrame(data_id, columns=[key]) # converts tags to DataFrame
        data[i] = pd.concat([data_id, data[i]], axis=1) # affixes tags to data
        if 'index' in data[i].columns: # if the affixation creates a surplus index, delete it
            data[i] = data[i].drop(columns=['index'])

    # Merges data of different types (e.g. data from different runs)
    all_data = pd.concat(data[:], ignore_index=True)


    return all_data



def combine_names(data, keys, joining='&'):
    """ Concatenates names together, useful in the event that unique combinations of channels need to be found.
    
    PARAMETERS
    ----------
    data : pandas DataFrame
    -   A DataFrame containing the names of the channels.

    keys : list of strs
    -   The keys of the columns in the DataFrame to be used.

    joining : str
    -   A string with which to join multiple names.


    RETURNS
    ----------
    names : list of strs
    -   The list of combined channel names.
    """

    names = [] # get names of all channels
    for chann_i in data.index:
        cat_name = [] # if multiple channel names, combine them 
        for key in keys:
            cat_name.append(data[key][chann_i])
        cat_name = joining.join(cat_name)
        names.append(cat_name)


    return names



def unique_names(names):
    """ Finds the unique names.

    PARAMETERS
    ----------
    names : list of strs
    -   Names to find the unique entries of.
    

    RETURNS
    ----------
    unique_names : list of strs
    -   Unique names.


    unique_idc : list of ints
    -   Indices of each unique name.
    """

    unique_names = pd.unique(names) # find unique channel names
    unique_idc = [] # find indices of these unique channels
    for unique_name in unique_names:
        unique_idc.append([i for i, x in enumerate(names) if x == unique_name])


    return unique_names, unique_idc



def unique_channel_types(unique_ch_idc, types):
    """ Find the types of unique channels (e.g. those derived from unique_channel_names) in the data.

    PARAMETERS
    ----------
    unique_ch_idc : list of ints
    -   Indices of each unique channel, as derived from unique_channel_names.

    types : list of strs
    -   The types of all channels in the data.


    RETURNS
    ----------
    unique_types_names : list of strs
    -   The types of unique channels in the data.

    unique_types_idc : list of ints
    -   Indices of types of each unique channel in unique_ch_idc
    """

    unique_types_names = np.unique(types) # find unique types
    unique_types_idc = [] # find indices of unique types

    for unique_types_name in unique_types_names:
        unique_types_idc.append([])
        for ch_idc in unique_ch_idc:
            first = True
            for ch_idx in ch_idc:
                if types[ch_idx] == unique_types_name:
                    if first == True:
                        unique_types_idc[-1].append([])
                    unique_types_idc[-1][-1].append(ch_idx)
                    first = False


    return unique_types_names, unique_types_idc



def channel_reref_types(names):
    """ Find the indices of channels based on the type of rereferencing.

    PARAMETERS
    ----------
    names : list
    -   Names of the channels to group.


    RETURNS
    ----------
    reref_types : dict
    -   A dictionary in which the keys are the type of rereferencing and values are the indices of the channels.


    NOTES
    ----------
    -   Only supports CAR and bipolar rereferencing.
    """

    reref_types = {'CAR': [],
                   'bipolar': []}
    
    for i, name in enumerate(names):
        if 'CAR' in name:
            reref_types['CAR'].append(names.index[i])
        else:
            reref_types['bipolar'].append(names.index[i])

    return reref_types



def index_by_type(data, info, avg_as_equal=False):
    """ Find the indices of channels based on the type of rereferencing.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data to organise.

    info : dict
    -   A dictionary containing information about the data characteristics, with keys representing the characteristics
        and the corresponding values reflecting the unique types of data for that characteristic.
    
    avg_as_equal : bool, default False
    -   Whether or not data that has been averaged should be treated as the same type, regardless of what has been
        averaged. If False (default), the data is not necessarily treated as the same type (and thus is assigned a
        colour), whereas if True, the data is treated as the same type (and thus not assigned a colour).
    -   E.g. if some data has been averaged across subjects 1 and 2, whilst other data has been averaged across subjects
        3 and 4, avg_as_equal=True means that this data will be considered equivalent (and thus not assigned a colour)
        based on the fact that it has been averaged, regardless of the fact that it was averaged across different
        subjects (which can be useful if you want to generate contrast between data based on their different
        characteristics, rather than diluting the colours with their similarities).


    RETURNS
    ----------
    idcs : list of lists
    -   List containing sublists of the data indices sharing the same charactertistics.
    """

    if avg_as_equal == True:
        for group in info.keys():
            for i, val in enumerate(data[group]):
                if 'avg' in val:
                    data[group][i] = 'avg'
            for i, val in enumerate(info[group]):
                if 'avg' in val:
                    info[group][i] = 'avg'
            info[group] = np.unique(info[group])


    comb_types = ['-' for i in range(np.shape(data)[0])]
    for group_key in info.keys():
        if len(info[group_key]) > 1:
            for i, val in enumerate(data[group_key]):
                comb_types[i] += val
    unique_comb_types = np.unique(comb_types)

    idcs = []
    for unique_comb_type in unique_comb_types:
        idcs.append([])
        for i, val in enumerate(comb_types):
            if val == unique_comb_type:
                idcs[-1].append(data.index[i])


    return idcs



def same_axes(data, border=2, floor=None):
    """ Uses the maximum and minimum values in a dataset to create a single axis limit for this data.

    PARAMETERS
    ----------
    data : pandas DataFrame or list
    -   The data to set an axis limit for.

    border : int | float
    -   The amount (percentage of data range) to add to the axis limits.

    floor : int | float | None
    -   The lower limit of the axis. If None, the limit is set based on the minimum value in the data. If a number, that
        is the limit.


    RETURNS
    ----------
    lim : list of size (1x2)
    -   A list specifying the lower (entry 0) and upper (entry 1) axis limits.
    """

    ## Setup
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) == True: # If data is in a DataFrame, flattens data into a 1D list
        data = [item for sublist in data.values.flatten() for item in sublist]
    elif isinstance(data, list) == True: # If data is in a list, no flattening is necessary
        data = data
    else: # Checks the correct datatype is given
        raise ValueError("The data must be given as a pandas DataFrame or a list.")
        

    # Finds the minimum and maximum values in the dataset
    lim = [0, 0]
    if floor == None:
        lim[0] = min(data)
    else:
        lim[0] = floor
    lim[1] = max(data)

    # If requested, adds a buffer to the axis limits
    if border:
        data_range = lim[1] - lim[0]
        border_val = data_range*(border/100)
        lim[0] -= border_val
        lim[1] += border_val


    return lim



def window_title(info, base_title='', full_info=True, alphabetical=True):
    """ Generates a title for a figure window based on information in the data.
    
    PARAMETERS
    ----------
    info : dict
    -   A dictionary with keys representing factors in the data (e.g. experimental conditions, subjects, etc...) and
        their corresponding values in the data. E.g. 'task' = ['Rest', 'Movement'], 'subject' = [1, 2, 3], etc...

    base_title : str | Default ''
    -   A starting string to build the rest of the title on.

    full_info : bool, default True
    -   Whether or not full information about the data should be provided for averaged results. If True (default), full
        information (e.g. the IDs of the runs, subjects, etc... that the data was averaged over). If False, only a
        generic 'avg' tag is returned, useful for avoiding clutter on figures.

    alphabetical : bool, default True
    -   Whether or not to reorganise the keys of info alphabetically. Useful for ensuring consistency in titles.


    RETURNS
    ----------
    title : str
    -   A title for the figure window based on information in the data. Factors are included in the title if all the
        plots for this data contain the same information.
    -   E.g. if all data to be plotted is from the same task, this information can be included in the window's title to
        prevent it from cluttering-up individual plots.

    included : list of strs
    -   A list of information that has been included in the title.
    -   This is useful if additional titles need to be generated (e.g. for subplots) so that you can avoid repeating
        information that has already been included in the window title.
    """

    ## Setup
    if full_info == False: # if partial information is requested, provide a general 'avg' tag rather than a tag...
    #... containing details of each piece of data involved
        for key in info.keys():
            for val_i, val in enumerate(info[key]):
                if val[:3] == 'avg': # if the data has been averaged, discard any additional info
                    info[key][val_i] = 'avg'
            info[key] = pd.unique(info[key])

    if alphabetical == True: # rearrange keys of info alphabetically, if requested
        new_info = {}
        for key in sorted(info):
            new_info[key] = info[key]
        info = new_info

    ## Title creation
    title = base_title
    included = []
    for key in info.keys():
        if len(info[key]) == 1: # if only a single type of data is present, add this information to the window's title
            title += f'{key}-{info[key][0]},'
            included.append(key)
    if included != []:
        title=title[:-1] # removes the extra comma at the end of the title


    return title, included



def plot_title(info, already_included=[], base_title='', full_info=True, alphabetical=True):
    """ Generates a title for a subplot based on information in the data.
    
    PARAMETERS
    ----------
    info : dict
    -   A dictionary with keys representing factors in the data (e.g. experimental conditions, subjects, etc...) and
        their corresponding values in the data. E.g. 'task' = ['Rest', 'Movement'], 'subject' = [1, 2, 3], etc...

    already_included : list of strs
    -   A list containing the information that has already been included in a previous title (i.e. the title of the
        window in which the subplot is contained). Empty by default.

    base_title : str
    -   A starting string to build the rest of the title on.
    
    full_info : bool, default True
    -   Whether or not full information about the data should be provided for averaged results. If True (default), full
        information (e.g. the IDs of the runs, subjects, etc... that the data was averaged over). If False, only a
        generic 'avg' tag is returned, useful for avoiding clutter on figures.
    
    alphabetical : bool, default True
    -   Whether or not to reorganise the keys of info alphabetically. Useful for ensuring consistency in titles.


    RETURNS
    ----------
    title : str
    -   A title for the channel subplot based on information in the data. Factors are included in the title if all the
        data for this channel contain the same information.
    -   E.g. if all data to be plotted is from the same task, this information can be included in the subplot's title to
        prevent it from cluttering-up the subplot's legend.

    included : list of strs
    -   A list of information that has been included in the title. This is useful if additional titles need to be
        generated (e.g. for data labels) so that you can avoid repeating information that has already been
        included in the window title.
    """

    ## Setup
    if full_info == False: # If partial information is requested, provide a general 'avg' tag rather than a tag...
    #... containing details of each piece of data involved
        for key in info.keys():
            for val_i, val in enumerate(info[key]):
                if val[:3] == 'avg': # if the data has been averaged, discard any additional info
                    info[key][val_i] = 'avg'
            info[key] = pd.unique(info[key])

    if alphabetical == True: # rearrange keys of info alphabetically, if requested
        new_info = {}
        for key in sorted(info):
            new_info[key] = info[key]
        info = new_info

    ## Title creation
    title = base_title
    included = []
    for key in info.keys():
        if len(info[key]) == 1 and key not in already_included: # if only a single type of data is present and this...
        #... information has not been included in the window's title, add this information to the channel's title
            if len(title) >= 30: # adds a new line to the title to stop it getting too wide
                title += '\n'
            title += f'{key}-{info[key][0]},'
            included.append(key)
    if included != []:
        title=title[:-1] # removes the extra comma at the end of the title


    return title, included



def data_title(info, already_included=[], full_info=True, alphabetical=True):
    """ Generates a title for a subplot based on information in the data.
    
    PARAMETERS
    ----------
    info : dict
    -   A dictionary with keys representing factors in the data (e.g. experimental conditions, subjects, etc...) and
        their corresponding values in the data. E.g. 'task' = ['Rest', 'Movement'], 'subject' = [1, 2, 3], etc...

    already_included : list of strs
    -   A list containing the information that has already been included in a previous title (i.e. the title of the
        window in which the subplot is contained). Empty by default.

    base_title : str
    -   A starting string to build the rest of the title on.

    full_info : bool, default True
    -   Whether or not full information about the data should be provided for averaged results. If True (default), full
        information (e.g. the IDs of the runs, subjects, etc... that the data was averaged over). If False, only a
        generic 'avg' tag is returned, useful for avoiding clutter on figures.

    alphabetical : bool, default True
    -   Whether or not to reorganise the keys of info alphabetically. Useful for ensuring consistency in titles.

    RETURNS
    ----------
    title : str
    -   A label for the data based on the data's information. Factors are included in the label if this information
        cannot be placed in the subplot's or figure window's title (i.e. if it is different for the data being plotted).
    """

    ## Setup
    if full_info == False: # If partial information is requested, provide a general 'avg' tag rather than a tag...
    #... containing details of each piece of data involved
        for key in info.keys():
            for val_i, val in enumerate(info[key]):
                if val[:3] == 'avg': # if the data has been averaged, discard any additional info
                    info[key][val_i] = 'avg'
            info[key] = pd.unique(info[key])

    if alphabetical == True: # rearrange keys of info alphabetically, if requested
        new_info = {}
        for key in sorted(info):
            new_info[key] = info[key]
        info = new_info

    ## Title creation
    title = ''
    first = True
    for key in info.keys():
        if key not in already_included:  # if only a single type of data is present and this...
        #... information has not been included in the window's and subplot's titles, add this information to the data...
        #... label
            title += f'{key}-{info[key][0]},'
            first = False
    if first == False:
        title=title[:-1] # removes the extra comma at the end of the title


    return title



def get_colour_info(data, keys):
    """ Gets information about the data to feed into data_colour to determine the colour of data when plotting.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   A DataFrame containing the data to analyse.

    keys : list of strs
    -   The keys of the columns in the DataFrame to use to determine the colour of the data.


    RETURNS
    ----------
    info : dict
    -   A dictionary whose keys are the data characteristics requested and whose values consist of a 'binary' or
        'non-binary' marker (reflecting the nature of the data) followed by the unique values of the data.
    """

    binary_info = ['med', 'stim', 'data_type', 'ch_type', 'method']
    info = {}
    for key in keys:
        info[key] = [0,0]
        if key in binary_info:
            info[key][0] = 'binary'
        else:
            info[key][0] = 'non-binary'
        info[key][1] = list(np.unique(data[key]))

    return info



def data_colour(info, not_for_unique=False, avg_as_equal=False):
    """ Randomly generates colours for the data based on the data's characteristics.

    PARAMETERS
    ----------
    info : dict
    -   A dictionary containing details about the characteristics of the data (e.g. the tasks, subjects, etc...). Each
        key represents a different characteristic, with each value being a list whose first entry is the string 'binary'
        or 'non-binary' (reflecting whether there are only two possible types of data (binary) or more (non-binary)) and
        whose second entry is a list containing the types of data present.
    -   E.g. if the dataset contains data from a
        condition in which medication is present and a condition in which it is not, the corresponding key-value pair
        could be 'medication': ['binary', ['On', 'Off']]]. E.g. if the dataset contains data from multiple subjects, the
        corresponding key-value pair could be 'subjects': ['non-binary', ['01', '02', '03]].

    not_for_unique : bool, default False
    -   Whether or not colours should be generated for data characteristics that only have one type. If False (default),
        colours are generated for this data, whereas if True, colours are not generated for this data.
    -   E.g. if all the data is from the same medication condition (e.g. MedOn), then not_for_unique=True would mean
        this data is not assigned a colour, which can be useful if you want to generate contrast between data based on
        their different  characteristics (e.g. data from the same medication condition, but from different subjects).

    avg_as_equal : bool, default False
    -   Whether or not data that has been averaged should be treated as the same type, regardless of what has been
        averaged. If False (default), the data is not necessarily treated as the same type (and thus is assigned a
        colour), whereas if True, the data is treated as the same type (and thus not assigned a colour).
    -   E.g. if some data has been averaged across subjects 1 and 2, whilst other data has been averaged across subjects
        3 and 4, avg_as_equal=True means that this data will be considered equivalent (and thus not assigned a colour)
        based on the fact that it has been averaged, regardless of the fact that it was averaged across different
        subjects (which can be useful if you want to generate contrast between data based on their different
        characteristics, rather than diluting the colours with their similarities).
    

    RETURNS
    ----------
    colours : dict
    -   A dictionary containing the colours for the data, organised into keys representing the data characteristics and
        associated values for the particular colour of the data type within these characteristics.
    -   E.g. if there are 3 subjects, a corresponding key-value pair could be 'subjects': [[rgb triplet 1],
        [rgb triplet 2], [rgb triplet 3]].
    
    
    NOTES
    ----------
    -   If data characteristics in info are marked as binary, a colour is generated randomly for one type and the
        inverse colour is generated for the other type.
    -   If data characteristics in info are marked as non-bonary, the colours are generated randomly for each type.
    """

    ## Setup
    skip_key = []
    for key in info.keys():
        # Treats averaged data as if it is the same (i.e. gives it the same colour regardless of differences over...
        #... what runs, subjects, etc... data was averaged over)
        if avg_as_equal == True:
            key_info = []
            for val in info[key][1]:
                key_info.append(val[:3])
            key_info = np.unique(key_info)
            if len(key_info) == 1 and key_info[0] == 'avg':
                skip_key.append(key)
        # Does not assign colours based on characterstics that do not differ between the data being plotted, so that...
        #... only the characteristics which are different determine the data's colour which makes the colours stand out
        if not_for_unique == True:
            if len(info[key][1]) == 1:
                skip_key.append(key)
    skip_key = np.unique(skip_key)


    ## Gets colours
    np.random.seed(seed=0) # ensures consistent, replicable plot colours
    colours = {}
    for key in info.keys():
        if key not in skip_key:
            if key != 'data_type':
                colours[key] = []
                first = True
                if info[key][0] == 'binary': # if only 2 types of data can be present, generate inverse colours for these types
                    if first == True:
                        first_val = info[key][1][0] # assigns the first type present to be the default type
                        first_val_colour = np.random.rand(1, 3) # sets the colour for this type
                        second_val_colour = 1 - first_val_colour # sets the colour for the other type as the inverse
                    for char in info[key][1]: # sets the colour for the two types
                        if char == first_val:
                            colours[key].append(first_val_colour)
                        else:
                            colours[key].append(second_val_colour)
                        colours[key][-1] = np.append(colours[key][-1],1) # adds an alpha value
                elif info[key][0] == 'non-binary': # if many types of data can be present, randomly generate colours
                    for x in info[key][1]:
                        colours[key].append(np.random.rand(1, 3))
                        colours[key][-1] = np.append(colours[key][-1],1) # adds an alpha value
                else:
                    raise ValueError(f'The colour key {info[key][0]} is not recognised. Only binary and non-binary are accepted.')
            else:
                colours[key] = []
                for char in info[key][1]:
                    if char == 'real':
                        alpha = 1
                    elif char == 'shuffled':
                        alpha = .3
                    else:
                        raise ValueError(f"Only the data types 'real' and 'shuffled' are recognised, but {char} is present.")
                    colours[key].append(np.asarray((*[np.nan]*3, alpha)))


    return colours



def save_fig(fig, filename, filetype, directory='figures', foldername=''):
    """ Saves plotted figures.

    PARAMETERS
    ----------
    fig : matplotlib Figure object
    -   The figure to be saved.

    filename : str
    -   The name of the file to save the figure as.

    filetype : str
    -   The filetype to save the image as (e.g. 'png', 'jpg'). No fullstop is necessary (e.g. incorrect: '.png',
        correct: 'png').
    
    directory : str
    -   The location where the images should be saved.

    foldername : str
    -   If given (default an empty string), images are saved in this folder within the directory (the folder is created
        if it does not already exist).

    
    RETURNS
    ----------
    N/A


    NOTES
    ----------
    -   This function should be called after the figures have already been shown and manipulated (e.g. the window
        maximised to achieve the desired layout of the subplots and text).
    """

    dir_folder = os.path.join(directory, foldername) # folder directory

    # Makes sure the filename is legal and replaces any illegal characters with a semicolon
    legal_name = ''
    for char in filename:
        if char in "\/:*?<>|": # if character is illegal...
            char = ';' #... replace with a legal character
        legal_name += char
    filename = legal_name

    # Creates the folder, if requested
    if foldername != '':
        try: # if the folder doesn't exist, make a new one,...
            os.mkdir(dir_folder)
        except: #... otherwise, do nothing
            dir_folder = dir_folder

    # Creates the file
    curr_files = os.listdir(dir_folder) # gets files in the directory
    if curr_files != []:
        n_copies = 0 # the number of files in the directory sharing the same name
        copy = True # whether there are multiple files with the same name
        first = True # whether this is the first instance of the filenames repeating
        while copy == True:
            if first == True:
                if f'{filename}.{filetype}' in curr_files: # checks if the filename is already being used
                    n_copies += 1 # notes the number of files sharing this name
                    first = False
                    if f'{filename}({str(n_copies)}).{filetype}' in curr_files: # if a file is using the next...
                    #... available name (e.g. filename(1).png), further increase the number of files sharing names
                        n_copies += 1
                    else: # if not, use the name filename(1).png
                        copy = False
                        break
                else: # if the filename is not being used, no need to change anything
                    copy = False
                    first = False
                    break
            else: # checks if the modified name (e.g. filename(2).png) is in use...
                if f'{filename}({str(n_copies)}).{filetype}' in curr_files: #... if so, increment n_copies to check...
                #... for the next available name,
                        n_copies += 1
                else: #... otherwise, use the modified filename
                    copy = False
                    break
        if n_copies != 0: # sets the filename so that it is unique and not overwriting previous files
            filename += f'({str(n_copies)})'
    
    # Gets the complete filename and save directory
    filename += f'.{filetype}'
    dir_file = os.path.join(dir_folder, filename)

    # Saves the figure
    fig.savefig(dir_file, bbox_inches='tight')
    print(f"Saving figure as {dir_file}")

    

def check_identical(data, halt_script=True):
    """ Checks to see if multiple sets of data are identical.

    PARAMETERS
    ----------
    data : array
    -   A list containing the data to be compared.

    halt_script : bool, default True
    -   Whether or not to raise an error and stop the script if the data is not identical. If True (default), an error
        is raised and the script stops. If False, the script is not stopped, but the non-identical nature of the data is
        returned and a warning is raised.
    

    RETURNS
    ----------
    identical : bool
    -   Whether or not the data is identical. If True, the data is identical. If False, the data is not identical.
    """

    identical = True
    stop = False
    if len(data) > 1: # if there are multiple sets of values to compare
        sample_1 = data[0] # the first set of values against which all others will be compared
        if stop == False:
            for sample_2 in data[1:]: # for the second set of values onwards...
                result = sample_1 == sample_2 #... check if the values are equal
                if type(result) == bool:
                    if result == False:
                        identical = False
                else:
                    if any(result) == False:
                        identical = False
                if identical == False:
                    if halt_script == True:
                        stop = True
                        print('The values for the data do not match, and therefore should not be averaged over.')
                        break
                    else:
                        raise ValueError(f'The values for the data do not match, and therefore the data cannot be averaged over.')
    

    return identical



def average_data(data, axis=0):
    """ Averages data and returns the standard deviation.

    PARAMETERS
    ----------
    data : array
    -   The data to average.

    axis : int
    -   The axis of the data to average over.


    RETURNS
    ----------
    avg : list
    -   The averaged data.

    std : list
    -   The standard deviation of the data that was averaged.

    sem : list
    -   The standard error of the mean of the data that was averaged.

    ci : list
    -   The 95% confidence interval of the data that was averaged.
    """

    # Converts data to a workable type
    if type(data) != list:
        data = data.tolist()

    if axis == 0:
        n_points = np.size(data, axis=1)
    elif axis == 1:
        n_points = np.size(data, axis=0)
    else:
        raise Exception("The data should be a 2D object.")

    # Processes data
    avg = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    sem = st.sem(data, axis=axis)
    ci = []
    #for i in range(n_points):
    #    intervals = st.t.interval(0.95, np.size(data, axis=axis)-1, avg[i], std[i])
    #    ci.append(intervals[1] - avg[i])


    return avg, std, sem, ci



def average_dataset(data, avg_over, separate, x_keys, y_keys, cat=[], data_keys=[], recalculate_maxs=True):
    """ Averages data over specific factors (e.g. runs, subjects).

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data to average.

    avg_over : str
    -   The column name of the data to average over.

    separate : list of strs
    -   The name(s) of the column(s) in the DataFrame containing information so that the correct data entries can be
        averaged over.

    x_keys : list of strs
    -   The names of the columns in the DataFrame containing the keys whose values should be identical for the entries
        that are being averaged (i.e. the independent/control variables).

    y_keys : list of strs
    -   The names of the columns in the DataFrame whose values should be averaged.

    cat : list of strs
    -   The names of the columns in the DataFrame whose values should be concatenated into a single list.

    data_keys : list of strs
    -   The names of the columns in the DataFrame whose maximum values should be recalculated. Only used if
        'recalculate_maxs' == True.

    recalculate_maxs : bool, default True
    -   Whether or not to recalculate the maximum frequency-band values (i.e. fbands_max, fbands_fmax) based on the
        averaged data. If True, these values are recalculated (and corresponding standard deviation values are
        discarded, with new values not computed). If False, these values are not recalculated, and the average of the
        pre-existing maximum values is used (with new corresponding standard values calculated).


    RETURNS
    ----------
    new_data : pandas DataFrame
    -   The averaged data.
    """

    ### Setup
    # Finds data to average over
    names = combine_names(data, separate)
    _, unique_idc = unique_names(names)
    
    # Makes sure the x_keys each unique channel are identical
    for x_key in x_keys: # for each key whose values should be identical
        for unique_i in unique_idc: # for each unique channel
            identical = check_identical(list(data[x_key][unique_i])) # the values to compare
            if identical == False:
                raise ValueError(f'The {x_key} values for the data do not match, and therefore the data cannot be averaged over.')
    
    # Gets the keys of the data whose values should be combined
    comb_keys = [key for key in data.keys() if key not in avg_over and key not in separate and key not in x_keys and 
                 key not in y_keys]
    stats = ['std', 'sem', 'ci']
    stat_keys = []
    for stat in stats:
        for key in y_keys:
            stat_keys.append(f'{key}_{stat}')
    discard = []
    for comb_key in comb_keys: # prevents keys containing S.D. data from being combined, as this...
    #... needs to be recalculated
        if comb_key in stat_keys:
            discard.append(comb_key)
    if discard != []:
        comb_keys = [key for key in comb_keys if key not in discard]

    
    ### Processing
    # Averages across the y_keys for each unique channel
    all_avg = {}
    all_std = {}
    all_sem = {}
    all_ci = {}
    for y_key in y_keys:
        all_avg[y_key] = []
        all_std[f'{y_key}_std'] = []
        all_sem[f'{y_key}_sem'] = []
        all_ci[f'{y_key}_ci'] = []
        for unique_i in unique_idc:
            avg, std, sem, ci = average_data(data[y_key][unique_i])
            all_avg[y_key].append(avg)
            all_std[f'{y_key}_std'].append(std)
            all_sem[f'{y_key}_sem'].append(sem)
            all_ci[f'{y_key}_ci'].append(ci)
    avg = all_avg
    std = all_std
    sem = all_sem
    ci = all_ci

    # Adds new columns for std and ci data of each y_key (if not already present)
    for stat in stats:
        holder = np.repeat(np.nan, np.shape(data)[0])
        for y_key in y_keys:
            if f'{y_key}_{stat}' not in data.columns:
                y_key_pos = list(data.columns).index(y_key)
                data.insert(y_key_pos+1, y_key+f'_{stat}', holder)

    # Collates data for new DataFrame
    new_cols = []
    for key in data.columns:
        new_cols.append([])
        if key == avg_over or key in comb_keys: # the averaged-over factor and factors that are inadvertently...
        #... averaged over gets special names, specifying what was averaged over (e.g. run IDs, subject IDs, etc...)
            for unique_i in unique_idc:
                if key in cat:
                    new_entry = []
                    for entry in data[key][unique_i]:
                        new_entry.append(entry)
                    new_cols[-1].append(new_entry)
                else:
                    new_entry = np.unique(data[key][unique_i]).tolist()
                    new_entry = ','.join(new_entry)
                    new_cols[-1].append(f'avg[{new_entry}]')
        elif key in y_keys: # the values that were averaged get used in place of the original values
            new_cols[-1].extend(avg[key])
        elif key in x_keys or key in separate: # values which are identical across the averaged group, so can just be...
        #... copied from the original data
            for unique_i in unique_idc:
                new_cols[-1].append(data[key][unique_i[0]])
        elif 'std' in key: # the std values (should not be in y_keys, as you don't want to average over std)
            new_cols[-1].extend(std[key])
        elif 'sem' in key: # the sem values (should not be in y_keys, as you don't want to average over std)
            new_cols[-1].extend(sem[key])
        elif 'ci' in key: # the ci values (should not be in y_keys, as you don't want to average over cis)
            new_cols[-1].extend(ci[key])
        else:
            raise ValueError(f"Unknown key '{key}' present when averaging data.")
    
    # Creates new DataFrame
    new_data = list(zip(*new_cols[:]))
    new_data = pd.DataFrame(data=new_data, columns=list(data.columns))

    # Recalculates maximum values based on the averaged data, if requested
    if recalculate_maxs == True:
        for data_key in data_keys:
            for data_i in new_data.index:
                data = new_data.iloc[data_i]
                recalc_max, recalc_fmax = recalculate_band_max(data[data_key], data['freqs'], data['fbands'])
                data['fbands_max'] = recalc_max
                data['fbands_fmax'] = recalc_fmax
            new_data = new_data.drop(columns=[data_key+'_fbands_max_std', data_key+'_fbands_fmax_std',
                                              data_key+'_fbands_max_sem', data_key+'_fbands_fmax_sem',
                                              data_key+'_fbands_max_ci', data_key+'_fbands_fmax_ci'])

            
    return new_data



def alter_by_condition(data, cond, types, method, separate, x_keys, y_keys, avg_as_equal=False, recalculate_maxs=True):
    """ Alters the data of different types for a particular condition group that are otherwise identical

    PARAMETERS
    ----------
    data : Pandas DataFrame
    -   The data to alter.

    cond : string
    -   The key of the condition to use to alter the data based on different subtypes of this condition (e.g. med,
        stim).

    types : list of str
    -   Subtypes of the condition to use to alter the data (e.g. for a key of 'med', ['On', 'Off']). The order of the
        subtypes determines how the alteration occurs.

    method : str
    -   How to alter the data (e.g. 'subtract'). So far, only 'subtract' is supported.

    x_keys : list of strs
    -   The names of the columns in the DataFrame containing the keys whose values should be identical for the entries
        that are being altered (i.e. the independent/control variables). If they are not identical, a warning is raised
        and the value is set to NaN.

    y_keys : list of strs
    -   The names of the columns in the DataFrame whose values should be altered.

    avg_as_equal : bool, default False
    -   Whether or not data that has been averaged should be treated as the same type, regardless of what has been
        averaged. If False (default), the data is not necessarily treated as the same type (and thus is assigned a
        colour), whereas if True, the data is treated as the same type (and thus not assigned a colour).
    -   E.g. if some data has been averaged across subjects 1 and 2, whilst other data has been averaged across subjects
        3 and 4, avg_as_equal=True means that this data will be considered equivalent (and thus not assigned a colour)
        based on the fact that it has been averaged, regardless of the fact that it was averaged across different
        subjects (which can be useful if you want to generate contrast between data based on their different
        characteristics, rather than diluting the colours with their similarities).

    recalculate_maxs : bool, default True
    -   Whether or not to recalculate the maximum frequency-band values (i.e. fbands_max, fbands_fmax) based on the
        altered data. If True, these values are recalculated (and corresponding standard deviation values are
        discarded, with new values not computed). If False, these values are not recalculated based on the altered data,
        and the altered pre-existing maximum values are used.


    RETURNS
    ----------
    new_data : pandas DataFrame
    -   The DataFrame of the altered data.


    NOTES
    ----------
    -   E.g. The input key='med', types=['On', 'Off'], method='subtract' would mean medOff data is subtracted from
        medOn data, whereas the input key='med', types=['Off', 'On'], method='subtract' would mean medOn data is
        subtracted from medOff data. 
    """

    ### Setup
    ## Error checking
    # Checks the inputs are in the correct format
    if type(cond) != str: # only one condition can be modified at a time
        raise ValueError(f'Only one key should be provided, but {len(cond)} are.')
    if method not in ['subtract']: # checks the correct method of alteration is provided
        raise ValueError(f"Only the method 'subtract' is supported.")
    if len(types) != 2: # makes sure the manipulation involves two groups
        raise ValueError(f"Two, and only two, subtypes of data for the condition should be provided, but {len(types)} is/are.")
    
    # Checks the inputs match the data entries
    if sorted(np.unique(data[cond])) != sorted(types):
        raise ValueError(f"All subtypes of data should be specified in the inputs, but only {types} is/are.")

    
    ### Processing
    # Changes the names of averaged entries, if requested
    if avg_as_equal == True:
        for col in data.columns:
            for row_i in range(len(data)):
                if data[col][row_i][:3] == 'avg':
                    data[col][row_i] = 'avg'

    # Finds data to alter
    names = combine_names(data, separate)
    unique_groups, unique_idc = unique_names(names)

    # Makes sure the x_keys for each unique group are identical
    for x_key in x_keys: # for each key whose values should be identical
        for unique_i in unique_idc: # for each unique group
            identical = check_identical(list(data[x_key][unique_i])) # the values to compare
            if identical == False: # Set the non-matching values to NaN and raise a warning
                for i in unique_i:
                    data[x_key][i] = np.nan
                print(f'The {x_key} values for the data do not match, setting the values for this {x_key} to NaN.')

    # Removes indices where only one type of the condition is present
    discard = []
    for idx, unique_i in enumerate(unique_idc):
        if len(unique_i) != 2:
            print(f"Warning: The group {unique_groups[idx]} does not contain entries for both condition types {cond} {types}.\nThis data will be discarded.")
            discard.append(idx)
    unique_idc = [unique_idc[i] for i in range(len(unique_idc)) if i not in discard]

    # Removes the S.D. and CI columns from the data for certain alteration methods
    stats = ['std', 'sem', 'ci']
    stat_cols = []
    for name in data.columns:
        for stat in stats:
            if stat in name:
                stat_cols.append(name)
    if method in ['subtract']:
        data = data.drop(columns=stat_cols)

    # Alters the data
    altered = {}
    for y_key in y_keys:
        altered[y_key] = []
        for unique_i in unique_idc:
            if method == 'subtract':
                if len(unique_i) != 2:
                    raise ValueError(f"Two, and only two, subtypes of data for the condition should be provided, but {len(unique_i)} is/are.")
                altered[y_key].append(data.loc[unique_i[0]][y_key] - data.loc[unique_i[1]][y_key])

    # Collates data for new DataFrame
    new_cols = []
    for key in data.columns:
        new_cols.append([])
        if key == cond: # the condition which the alteration has been based on
            if method == 'subtract':
                for unique_i in unique_idc:
                    new_cols[-1].append(f"{data[cond][unique_i[0]]}-{data[cond][unique_i[1]]}")
        elif key in y_keys: # the values that were averaged get used in place of the original values
            new_cols[-1].extend(altered[key])
        elif key in x_keys or key in separate: # values which are identical across the averaged group, so can just be...
        #... copied from the original data
            for unique_i in unique_idc:
                new_cols[-1].append(data[key][unique_i[0]])
        else:
            print(f"Warning: Unknown key '{key}' present when altering data, setting as NaN.")
            new_cols[-1].extend([np.nan]*len(unique_idc))
    
    # Creates new DataFrame
    new_data = list(zip(*new_cols[:]))
    new_data = pd.DataFrame(data=new_data, columns=list(data.columns))

    # Recalculates maximum values based on the altered data, if requested
    if recalculate_maxs == True:
        # Checks for the correct format of the data
        if 'psd' in new_data.keys() and 'coh' not in new_data.keys():
            data_key = 'psd'
        elif 'coh' in new_data.keys() and 'psd' not in new_data.keys():
            data_key = 'coh'
        else:
            raise ValueError(f"One and only one of 'psd' or 'coh' must be included in the DataFrame.")
        
        # Recalculates maximum values
        for data_i in new_data.index:
            data = new_data.iloc[data_i]
            recalc_max, recalc_fmax = recalculate_band_max(data[data_key], data.freqs, data.fbands)
            new_data['fbands_max'].iloc[data_i] = recalc_max
            new_data['fbands_fmax'].iloc[data_i] = recalc_fmax
        if 'fbands_max_std' in new_data.columns:
            new_data = new_data.drop(columns=['fbands_max_std'])
        if 'fbands_fmax_std' in new_data.columns:
            new_data = new_data.drop(columns=['fbands_fmax_std'])
        if 'fbands_max_sem' in new_data.columns:
            new_data = new_data.drop(columns=['fbands_max_sem'])
        if 'fbands_fmax_sem' in new_data.columns:
            new_data = new_data.drop(columns=['fbands_fmax_sem'])
        if 'fbands_max_ci' in new_data.columns:
            new_data = new_data.drop(columns=['fbands_max_ci'])
        if 'fbands_fmax_ci' in new_data.columns:
            new_data = new_data.drop(columns=['fbands_fmax_ci'])
            


    return new_data



def freq_band_info(names=None):
    """ Dictionary of frequency bands and their corresponding frequencies (in Hz).

    PARAMETERS
    ----------
    names : list of str | None (default)
    -   The names of bands to be returned. If None (default), all bands are included in the output.


    RETURNS
    ----------
    bands : dict
    -   A dictionary consisting of key-value pairs for frequency band names and their corresponding frequencies (in Hz).
    """

    bands = {
        'theta':     [4, 8],
        'alpha':     [8, 12],
        'low beta':  [13, 20],
        'high beta': [20, 35],
        'all beta':  [13, 35],
        'gamma':     [60, 90]
    }

    # Returns requested bands, if appropriate
    keep_keys = [] # finds which bands should be kept
    if names != None:
        for name in names:
            if name in bands.keys():
                keep_keys.append(name)
            else:
                raise ValueError(f'The requested band {name} is not a recognised frequency band.')
    discard = [] # removes the unrequested bands
    for band in list(bands.keys()):
        if band not in keep_keys:
            bands.pop(band)


    return bands



def freq_band_indices(freqs, bands, include_outside=False):
    """ Finds the indices of frequency bands in data.

    PARAMETERS
    ----------
    freqs : list of int | list of float
    -   The frequencies that you want to index.

    bands : dict with values of shape (n_bands x 2)
    -   A dictionary containing the frequency bands to be indexed. Each value should have a length of 2, i.e. a lower
        and a higher frequency to index.

    include_outside : bool, default False
    -   Whether or not to index frequencies outside the requested bands in the events that the bands do not align with
        the available frequencies.
    -   E.g. Say a band of 3-5 was requested, but the frequencies are [0, 2, 4, 6, 8]), include_shuffled=False would not
        include the indices of frequency values outside the requested band (i.e. this function would return [2],
        corresponding to 4 Hz), whereas include_shuffled=True would include these extra indices (i.e would return
        [1, 2, 3], corresponding to 2, 4, and 6 Hz).


    RETURNS
    ----------
    idc : dict of shape (n_bands x 2)
    -   A dictionary containing values of the indices of the lower and upper frequency band limits.
    """

    keys = bands.keys()
    idc = deepcopy(bands)

    for key in keys:
        band_freqs = bands[key]
        if len(band_freqs) == 2:
            if band_freqs[0] < band_freqs[1]:
                if min(freqs) > band_freqs[1] or max(freqs) < band_freqs[0]: # if the frequency to index is outside...
                #... the range (too low or too high)
                    idc[key] = [np.nan, np.nan]
                    print(f'WARNING: The requested frequency band [{band_freqs[0]}, {band_freqs[1]}] is not contained in the data, setting indeces to NaN.')
                else:
                    if min(freqs) > band_freqs[0]: # if the minimum frequency available is higher than lower band...
                    #... range, change it
                        band_freqs[0] = min(freqs)

                    if max(freqs) < band_freqs[1]: # if the maximum frequency available is lower than the higher band...
                    #... range, change it
                        band_freqs[1] = max(freqs)

                    if include_outside == False: # only gets frequencies within the band
                        # finds value closest to, but equal to/higher than the requested frequency
                        idc[key][0] = int(np.where(freqs == freqs[freqs >= band_freqs[0]].min())[0])
                        # finds value closest to, but equal to/lower than the requested frequency
                        idc[key][1] = int(np.where(freqs == freqs[freqs <= band_freqs[1]].max())[0])
                    else: # gets frequencies outside the band
                        # finds value closest to, but equal to/lower than the requested frequency
                        idc[key][0] = int(np.where(freqs == freqs[freqs <= band_freqs[0]].max())[0])
                        # finds value closest to, but equal to/higher than the requested frequency
                        idc[key][1] = int(np.where(freqs == freqs[freqs >= band_freqs[1]].min())[0])
            else:
                raise ValueError('The first frequency value must be lower than the second value.')  
        else:
            raise ValueError(f'There should be two entries for the frequency band, not {band_freqs}.')


    return idc



def find_unique_shuffled(fullnames):
    """ Finds the names and indices of shuffled channels, discarding duplicates.

    PARAMETERS
    ----------
    fullnames : list of str
    -   The names of channels to be filtered.


    RETURNS
    ----------
    realnames : dict
    -   A dictionary with keys 'name' and 'idx'. 'name' contains the unique channel names (with the shuffled affix
        removed). 'idx' contains the indices of each unique shuffled channel in the variable fullnames.
    """
    
    # Finds the shuffled channels
    shuffled_idc = []
    for i, name in enumerate(fullnames):
        if name[:8] == 'SHUFFLED':
            shuffled_idc.append(i)

    # Removes the shuffled affix from the channel names, then removes duplicate channel names
    realnames = {'name': [], 'idx': []}
    for idx in shuffled_idc:
        realnames['name'].append(fullnames[idx][fullnames[idx].index('_')+1:])
    realnames['name'] = np.unique(realnames['name']).tolist()

    # Finds where these unique channels are present
    for i, name in enumerate(realnames['name']): # for each unique channel name
        realnames['idx'].append([])
        for idx in shuffled_idc: # for each shuffled channel
            if name in fullnames[idx]: # if the channel contains the unique channel name
                realnames['idx'][i].append(idx) # put the channel indexes of the same name together


    return realnames



def rename_shuffled(names, types):
    """ Removes the affix from shuffled channel names so that shuffled data from the same channels shares the same name
        (useful for later processing).
    
    PARAMETERS
    ----------
    names : list of strs
    -   The names of the channels in the data.

    types : list of strs
    -   The types of data in the channels (i.e. 'real' or 'shuffled') to ensure that only the shuffled channels' names
        are altered.

    
    RETURNS
    ----------
    names : list of strs
    -   The names of the channels in the data with the modified names for the shuffled data.    
    """

    for i, name in enumerate(names):
        if types[i] == 'shuffled':
            names[i] = name[name.index('_')+1:] # removes 'SHUFFLED-X_' from the channel name


    return names



def average_shuffled(data, keys_to_avg, group_keys):
    """ Averages data over adjacent shuffled channels of the same type.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data to average over. Must contain a column that can be used to identify the shuffled channels.

    keys_to_avg : list of strs
    -   The keys of the DataFrame columns to be averaged.

    group_keys : list of strs
    -   The names of the column used to group the data for averaging over.


    RETURNS
    ----------
    data : pandas DataFrame
    -   The data with entries from adjacent shuffled channels of the same type averaged over.
    """

    og_data = data.copy()

    # Removes data that isn't shuffled (i.e. real)
    remove = []
    for i, data_type in enumerate(data.data_type):
        if data_type != 'shuffled':
            remove.append(i)
    data = data.drop(remove)

    # Finds the indices of the shuffled channel groups
    names = combine_names(data, group_keys)
    names_shuffled, idcs_shuffled = unique_names(names)
    for i, idc_shuffled in enumerate(idcs_shuffled):
        idcs_shuffled[i] = [data.index[j] for j in idc_shuffled]

    # Averages shuffled data
    discard = [] # entries in og_data to discard (as will not contain averaged data)
    for idc_shuffled in idcs_shuffled:
        if len(idc_shuffled) > 1:
            discard.extend(idc_shuffled[1:])
            for key in keys_to_avg:
                og_data[key][idc_shuffled[0]] = np.mean(data[key][idc_shuffled].to_list(), axis=0)
    og_data = og_data.drop(discard) # deletes the redundant entries
    og_data = og_data.reset_index(drop=True)


    return og_data



def data_by_band(data, measures, band_names=None):
    """ Calculates band-wise values based on the frequency-wise data.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The frequency-wise data.

    measures : list of str
    -   The key(s) in data to calculate the band-wise values for.

    band_names : list of str | None (default)
    -   The frequency bands to calculate band-wise values for. If None (default), all frequency bands provided
        by the helpers.freq_band_info function are used.
    
    
    RETURNS
    ----------
    data : pandas DataFrame
    -   The original frequency-wise data, plus band-wise values. These values include: the average in each band (avg);
        the maximum in each band (max); and the frequency at which this maximum occurs (fmax).
    """

    # Gets the band-wise data
    bands = freq_band_info(band_names) # the frequency bands to analyse

    for measure_i, measure in enumerate(measures):
        band_data = {}
        band_data['bands'] = []
        band_data['avg'] = [] # average in each frequency band
        band_data['max'] = [] # maximum in each frequency band
        band_data['fmax'] = [] # frequency of maximum in each frequency band
        for data_i in range(len(data.index)):
            band_data['bands'].append(list(bands.keys()))
            band_data['avg'].append([])
            band_data['max'].append([])
            band_data['fmax'].append([])
            band_idc = freq_band_indices(data['freqs'][data_i], bands)
            for key in bands.keys():
                band_data['avg'][data_i].append(data.iloc[data_i][measure][band_idc[key][0]:band_idc[key][1]+1].mean())
                band_data['max'][data_i].append(data.iloc[data_i][measure][band_idc[key][0]:band_idc[key][1]+1].max())
                band_data['fmax'][data_i].append(data['freqs'][data_i][np.where(data.iloc[data_i][measure] == 
                                                                                band_data['max'][data_i][-1])[0]])
        
        # Collects band-wise data
        fbands_keys = [measure+'_fbands_avg', measure+'_fbands_max', measure+'_fbands_fmax']
        if measure_i == 0: # if this is the first data being added, include the frequency band names
            band_data = list(zip(band_data['bands'], band_data['avg'], band_data['max'], band_data['fmax']))
            columns = ['fbands', *fbands_keys]
        else: # if data has already been added, don't include the frequency band names again
            band_data = list(zip(band_data['avg'], band_data['max'], band_data['fmax']))
            columns = fbands_keys
        band_data = pd.DataFrame(data=band_data, columns=columns)

        # Collects frequency- and band-wise data
        data = pd.concat([data, band_data], axis=1)


    return data



def recalculate_band_max(data, freqs, fbands):
    """ Calculates the maximum values and the frequencies of the maximum values in frequency bands.

    PARAMETERS
    ----------
    data : list, array
    -   The data to calculate the maximum values for, with size 1 x n_frequencies.

    freqs : list, array
    -   The frequencies corresponding to the values in data, with size 1 x n_frequencies.

    fbands : list of str
    -   The names of the frequency bands in which the maximum values should be calculated, with size 1 x
        n_frequency-bands.
    
    RETURNS
    ----------
    fbands_max : numpy array
    -   The maximum frequency values in each band, with size 1 x n_frequency-bands

    fbands_fmax : numpy array
    -   The frequency at which the maximum frequency values in each band occurs, with size 1 x n_frequency-bands.

    """

    fbands_max = []
    fbands_fmax = []
    fbands_info = freq_band_info(fbands) # the frequency bands to analyse
    fbands_idc = freq_band_indices(freqs, fbands_info) # indices of the frequency bands

    for fband in fbands:
        fbands_max.append(data[fbands_idc[fband][0]:fbands_idc[fband][1]+1].max()) # maximum frequency in each band
        fbands_fmax.append(freqs[np.where(data==fbands_max[-1])[0][0]]) # frequency of the maximum in each band

    return np.array(fbands_max), np.array(fbands_fmax)



def choose_aperiodic_mode(power, freqs, report=True, title=''):
    """ Log-log plots power data so that you can visually examine whether the aperiodic component of the FOOOF power
        model should be fitted in a fixed manner, or with a knee, additionally returning the result of this check, if
        requested.
    
    PARAMETERS
    ----------
    power : 1 x n list or array
    -   The power values, where n is the number of frequencies, corresponding to the values in 'freqs'.

    freqs : 1 x n list or array
    -   The n frequencies, corresponding to the values in 'power'.

    report : bool, default True
    -   Whether or not to prompt the user to input the fit mode that should be used. If 'True' (default), the user is
        asked to input the fit. If 'False', the user does not input the fit (i.e. this function is used only for
        visualising the log-log plotted PSD, and not for any further processing).

    title : str, default empty
    -   The title for the plotted power.

    RETURNS
    ----------
    fit : str
    -   The fit to use for the fitting the aperiodic component of the FOOOF model. Must be either 'fixed' or 'knee'.
        This is only returned if 'report' == True.

    """
    
    ### Error checking
    if type(report) != bool:
        raise ValueError(f"The variable 'report' must be a boolean, but it is a(n) {type(report)}.")


    ### Processing
    ## Plots the power
    plt.loglog(freqs, power)
    plt.title(title)

    ## Asks the user to choose what fit should be used, if requested
    if report == True:
        answered = False
        while not answered:
            fit = input("Which mode should be used for fitting the aperiodic component, 'fixed' or 'knee'?: ")
            if fit != 'fixed' and fit != 'knee':
                print('This is not a valid input. Try again.')
            else:
                answered = True
        
        return fit



def check_fm_fit(fm, report=True):
    """
    PARAMETERS
    ----------
    fm : FOOOF model object
    -   The FOOOF model whose fit should be examined.

    report : bool, default True
    -   Whether or not to prompt the user to judge whether the fit of the model is accetpable. If 'True' (default), the
        user is asked to choose whether the fit is acceptable. If 'False', the user is not asked to judge the data (i.e.
        this function is used only for visualising the fitted model and associated spectra).

    RETURNS
    ----------
    acceptable : bool
    -   Whether or not the FOOOF model fit is acceptable. This is only returned if report == 'True'.

    """

    ### Error checking
    if type(report) != bool:
        raise ValueError(f"The variable 'report' must be a boolean, but it is a(n) {type(report)}.")

    
    ### Processing
    ## Plots the model and associated spectra
    fig, axs = plt.subplots(2,1)
    # Periodic spectrum and model fit
    axs[0].set_title('Periodic')
    axs[0].plot(fm._spectrum_flat, linewidth=2, label='Periodic spectrum')
    axs[0].plot(fm._peak_fit, linestyle='--', linewidth=1.5, label='Periodic component model fit')
    axs[0].legend()
    # Aperiodic spectrum and model fit
    axs[1].set_title('Aperiodic')
    axs[1].plot(fm._spectrum_peak_rm, linewidth=2, label='Aperiodic spectrum')
    axs[1].plot(fm._ap_fit, linestyle='--', linewidth=1.5, label='Aperiodic component model fit')
    axs[1].legend()

    ## Asks the user to choose whether the fit is acceptable, if requested
    if report == True:
        answered = False
        while not answered:
            answer = input("Is the fit of the model acceptable, yes ('y') or no ('n')?: ")
            if answer == 'y':
                acceptable = True
                answered = True
            elif answer == 'n':
                acceptable = False
                answered = True
            else:
                print('This is not a valid input. Try again.')
        
        return acceptable



def check_fm_fits(power, freqs, params=None, title='', report=True):
    """ Fits the FOOOF models for the power spectra using both 'fixed' and 'knee' modes for fitting the aperiodic
        component, and plots PSDs and associated model fits so that you can decide which aperiodic mode should be used,
        with the option of inputing the desired mode which is then returned from the function.

    PARAMETERS
    ----------
    power : 1 x n list or array
    -   The power values, where n is the number of frequencies, corresponding to the values in 'freqs'.

    freqs : 1 x n list or array
    -   The n frequencies, corresponding to the values in 'power'.

    params : dict or None (default)
    -   Parameters to fit the model with. If None (default), the standard values are used, otherwise, the values in
        'params' are used.

    title : str, default empty
    -   The title for the plots, e.g. the channel name.

    report : bool, default True
    -   Whether or not to prompt the user to input the fit mode that should be used. If 'True' (default), the user is
        asked to input the fit and the associated model is returned. If 'False', the user does not input the fit (i.e.
        this function is used only for visualising the PSDs and model fits, and not for any further processing).

    RETURNS
    ----------
    aperiodic_mode : str
    -   The mode to use for the fitting the aperiodic component of the FOOOF model. Must be either 'fixed' or 'knee'.
        This is only returned if 'report' == True.
    
    model : FOOOF model object
    -   The model chosen by the user. This is only returned if 'report' == True.

    """

    ### Error checking
    if type(report) != bool:
        raise ValueError(f"The variable 'report' must be a boolean, but it is a(n) {type(report)}.")

    ### Sets parameters, if necessary
    standard_params = {'peak_width_limits': [0.5, 12],
                       'max_n_peaks': float('inf'),
                       'min_peak_height': 0,
                       'peak_threshold': 2}
    if params == None:
        params = standard_params
    else:
        for param in standard_params.keys():
            if param not in params.keys():
                params[param] = standard_params[param]

    
    ### Processing
    ## Generates the model fits
    # 'fixed' mode model
    fm_fixed = FOOOF(peak_width_limits=params['peak_width_limits'], max_n_peaks=params['max_n_peaks'],
                     min_peak_height=params['min_peak_height'], peak_threshold=params['peak_threshold'],
                     aperiodic_mode='fixed')
    fm_fixed.fit(freqs, power)
    # 'knee' mode model
    fm_knee = FOOOF(peak_width_limits=params['peak_width_limits'], max_n_peaks=params['max_n_peaks'],
                    min_peak_height=params['min_peak_height'], peak_threshold=params['peak_threshold'],
                    aperiodic_mode='knee')
    fm_knee.fit(freqs, power)

    ## Plots the PSDs and model fits
    fig, axs = plt.subplots(2,2)
    fig.suptitle(title)
    # 'fixed' mode model
    fm_fixed.plot(plt_log=False, ax=axs[0,0])
    fm_fixed.plot(plt_log=True, ax=axs[1,0])
    # 'knee' mode model
    fm_knee.plot(plt_log=False, ax=axs[0,1])
    fm_knee.plot(plt_log=True, ax=axs[1,1])

    ## Adds goodness of fit metrics to the plots
    gof_metrics = ['r_squared', 'error']

    # 'fixed' mode model
    fm_fixed_results = fm_fixed.get_results()
    fm_fixed_idc = [] # indices of the goodness of fit metrics in the results
    for metric in gof_metrics:
        fm_fixed_idc.append(fm_fixed_results._fields.index(metric))
    fm_fixed_gof = '' # string to print for the goodness of fit metrics
    for i, idx in enumerate(fm_fixed_idc):
        fm_fixed_gof += f"{gof_metrics[i]}={round(fm_fixed_results[idx], 3)}; "
    fm_fixed_gof = fm_fixed_gof[:-2] # removes the last semicolon and space
    axs[0,0].set_title(f"'fixed' aperiodic mode | {fm_fixed_gof}") # adds the info to the plot title

    # 'fixed' mode model
    fm_knee_results = fm_knee.get_results()
    fm_knee_idc = [] # indices of the goodness of fit metrics in the results
    for metric in gof_metrics:
        fm_knee_idc.append(fm_knee_results._fields.index(metric))
    fm_knee_gof = '' # string to print for the goodness of fit metrics
    for i, idx in enumerate(fm_knee_idc):
        fm_knee_gof += f"{gof_metrics[i]}={round(fm_knee_results[idx], 3)}; "
    fm_knee_gof = fm_knee_gof[:-2] # removes the last semicolon and space
    axs[0,1].set_title(f"'knee' aperiodic mode | {fm_knee_gof}") # adds the info to the plot title

    ## Prompts the user to decide on a fit and returns the choice, if requested
    if report == True:
        answered = False
        while not answered:
            aperiodic_mode = input("Which mode should be used for fitting the aperiodic component, 'fixed' or 'knee'?: ")
            if aperiodic_mode == 'fixed':
                model = fm_fixed
                answered = True
            elif aperiodic_mode == 'knee':
                model = fm_knee
                answered = True
            else:
                print('This is not a valid input. Try again.')
        
        plt.close()
        
        return model, aperiodic_mode



def collect_fm_info(fm):
    """ Collects information about the FOOOF model into a single dict.

    PARAMETERS
    ----------
    fm : FOOOF model object
    -   The model whose information should be collected

    RETURNS
    ----------
    info : dict
    -   A dictionary containing information about the FOOOF model.
    
    """

    ### Processing
    info_fields = {'meta_data': ['freq_range', 'freq_res'],
                   'results': ['aperiodic_params', 'peak_params', 'r_squared', 'error', 'gaussian_params'],
                   'settings': ['peak_width_limits', 'max_n_peaks', 'min_peak_height', 'peak_threshold',
                                'aperiodic_mode']}

    info = {}
    ## Meta data
    meta_data = fm.get_meta_data()
    field_idc = []
    for field in info_fields['meta_data']:
        field_idc.append(meta_data._fields.index(field))
    for i, idx in enumerate(field_idc):
        info[info_fields['meta_data'][i]] = meta_data[idx]
    
    ## Results
    results = fm.get_results()
    field_idc = []
    for field in info_fields['results']:
        field_idc.append(results._fields.index(field))
    for i, idx in enumerate(field_idc):
        info[info_fields['results'][i]] = results[idx]

    ## Settings
    settings = fm.get_settings()
    field_idc = []
    for field in info_fields['settings']:
        field_idc.append(settings._fields.index(field))
    for i, idx in enumerate(field_idc):
        info[info_fields['settings'][i]] = settings[idx]
            

    return info



def extract_data(data, select_type={}, extract={}):
    """ Extracts particular columns and rows from the a DataFrame to create a new DataFrame.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data from which particular entries will be taken.

    select_type : dict of lists
    -   A dictionary whose keys correspond to columns in the data and whose values (given as a list) correspond to the
        particular entries of those columns which should be extracted. E.g. 'med': ['on'], 'subject': [001, 002]. If not
        provided, all rows of data are taken.
    
    extract : dict
    -   A dictionary whose keys respond to columns in the data which should be included in the new DataFrame, and whose
        values correspond to the name(s) of this/these new column(s). If the value is None, the key is used as the
        column name (i.e. the same name as in the original DataFrame). If the value is a str, the str is used as the new
        columns name. If the value is a list (should be a list of strs), data in the original column will be extracted
        into several new columns. In this case, the length of the list which specifies the names of the new columns
        should match the length of the data in the original column. E.g. 'ch_coords' : ['x', 'y', 'z'] would put the
        first entry of 'ch_coords' into a new column 'x', the second entry into 'y', and the third entry into 'z'.

    RETURNS
    ----------
    data : pandas DataFrame
    -   The extracted data.
    
    """

    if select_type == {} and extract == {}:
        ### Error checking
        print(f"WARNING: No data is being extracted from the DataFrame, so the original data is being returned.")

    if select_type != {}:
        ### Discards the unwanted data
        remove = []
        for data_i in range(len(data)):
            for key in select_type.keys():
                if data.iloc[data_i][key] not in select_type[key]:
                    remove.append(data_i)
        remove = np.unique(remove)
        data = data.drop(remove)
        data = data.reset_index(drop=True)

    if extract != {}:
        ### Extracts the required data
        ## Sets up the new data holder
        extracted = {}
        for key in extract.keys():
            if extract[key] == None: # if the same name should be used for the column in the old and new dataset
                extracted[key] = data[key]
            elif type(extract[key]) == str: # if a different name should be used for the column in the old and new dataset
                extracted[extract[key]] = data[key]
            elif type(extract[key]) == list: # if data from a single columns in the old dataset is to be extracted into...
            #... multiple columns in the new dataset
                for subkey in extract[key]: # sets up the new columns
                    extracted[subkey] = []
                for data_i in range(len(data)):
                    data_x = data.iloc[data_i]
                    if len(data_x[key]) != len(extract[key]): # checks that the data is being extracted into the correct...
                    #... number of columns
                        raise ValueError(f"{len(data_x[key])} element(s) of data is/are attemtping to extract to {len(extract[key])} column(s).")
                    else:
                        for key_i, subkey in enumerate(extract[key]):
                            extracted[subkey].append(data_x[key][key_i])
    
    if select_type != {} or extract != {}:
        ## Creates the new DataFrame
        data = pd.DataFrame.from_dict(extracted)
    
    return data



def manipulate_data(data, calculate):
    """ Performs manipulations on data in a DataFrame.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data to be manipulated.

    calculate : dict of dicts
    -   Instructions on how to manipulate the data. The keys in calculate are the new columns of data that will be added
        to the DataFrame. Each entry of calculate is also a dictionary, with the following keys recognised:
        'col' : str (REQUIRED)
        -   The column in data that will be used for the manipulation.
        'method' : str (REQUIRED)
        -   The type of manipulation that will occur. 'avg', 'max', and 'min' are currently supported.
        'entries' : list of indices (OPTIONAL)
        -   The index of the values in the data that will be analysed. If not provided, all entries are analysed.
        'group_by' : list of str (OPTIONAL)
        -   Specifies the columns in data which should be used to identify the indices of data of the same type (e.g.
            the same subject and medication condition for ['subject', 'med']) which will be analysed together. If not
            provided, the manipulations are performed independently for each row of data. If the method requested is
            'max' or 'min', the corresponding value (i.e. highest or lowest value, respectively) is marked with a 1,
            whereas all other entries for this group are 0. If the method requested is 'avg', all rows for this group
            have the same value (i.e. the average value across this group)
        'drop_after' : bool (OPTIONAL)
        -   This indicates whether the column specified in 'col' should be removed from the DataFrame after the
            manipulation has occurred. If True, the column is removed. If False or if not provided, the columns is not
            removed.

    RETURNS
    ----------
    data : pandas DataFrame
    -   The manipulated data.
    
    """

    ### Performs manipulations on the data and adds new data to the DataFrame
    drop_cols = []
    for calc_key in calculate.keys():
        curr_calc = calculate[calc_key]
        new_entries = {}
        new_entries[calc_key] = []
        if 'drop_after' in curr_calc.keys():
            if curr_calc['drop_after'] == True:
                drop_cols.append(curr_calc['col'])
            
        if 'group_by' in curr_calc.keys():
            # Gets indices of grouped data
            names = combine_names(data, curr_calc['group_by'])
            names_group, idcs_group = unique_names(names)
            # Collects the data together and performs the calculation
            for idc in idcs_group:
                new_data = [0]*len(idc)
                curr_data = data.iloc[idc][curr_calc['col']]
                if curr_calc['method'] == 'max':
                    new_data[np.argmax(curr_data)] = 1
                elif curr_calc['method'] == 'min':
                    new_data[np.argmin(curr_data)] = 1
                elif curr_calc['method'] == 'avg':
                    new_data = [np.mean(curr_data) for x in new_data]
                else:
                    raise ValueError(f"The requested method {curr_calc['method']} is not supported.")
                new_entries[calc_key].extend(new_data)

        else:
            for data_i in range(len(data)):
                data_x = data.iloc[data_i]
                if 'entries' in curr_calc.keys():
                    curr_data = data_x[curr_calc['col']][curr_calc['entries']]
                else:
                    curr_data = data_x[curr_calc['col']]
                if curr_calc['method'] == 'max':
                    calculated = np.max(curr_data)
                elif curr_calc['method'] == 'min':
                    calculated = np.min(curr_data)
                elif curr_calc['method'] == 'avg':
                    calculated = np.mean(curr_data)
                else:
                    raise ValueError(f"The requested method {curr_calc['method']} is not supported.")
                new_entries[calc_key].append(calculated)

        ## Adds the new data
        new_data = pd.DataFrame.from_dict(new_entries)
        data = pd.concat([data, new_data], axis=1)

    ### Removes redundant columns before returning the data, if applicable
    if drop_cols != []:
        drop_cols = np.unique(drop_cols)
        data = data.drop(columns=drop_cols)

    return data



def add_loc_groups(data, ch_groups, ch_name_key="ch_name", ch_coords_key="ch_coords", handle_missing="error"):
    """ Adds channel localisation groups to the data.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   DataFrame with the names of the channels whose localisation groups are to be added, and their coordinates.

    ch_groups : pandas DataFrame
    -   DataFrame containing the names of the channels, their coordinates, and their localisation groups.

    ch_name_key : str (default "ch_name")
    -   The name of the key of channel names in 'data'. Default "ch_name".

    ch_coords_key : str (default "ch_coords")
    -   The name of the key of channel coordinates in 'data'. Default "ch_coords".

    handle_missing : str (default "warning")
    -   How to deal with channels in 'data' which do not have a corresponsing entry in 'ch_groups'. If "warning"
        (default), a warning about this is raised, but the script continues, and the groups of these channels is
        designated as "unassigned". If "error", an error about this is raised and the script stops.

    RETURNS
    ----------
    data : pandas DataFrame
    -   The original 'data' with the localisation groups added.
 
    """

    ### Error checking
    if handle_missing != "error" and handle_missing != "warning":
        raise ValueError(f"The requested method for 'handle_missing' {handle_missing} is not recognised. Only 'error' and 'warning' are accepted.")


    ### For each entry in 'data', assign the localisation group
    loc_groups = []
    for i in range(len(data)):
        entry = data.iloc[i]
        ## Finds the corresponding 'ch_groups' entry
        if type(ch_groups) != pd.DataFrame:
            if ch_groups == 'all':
                loc_groups.append('all')
        else: 
            if entry[ch_name_key] in ch_groups['ch_name'].values:
                ch_groups_i = np.where(ch_groups['ch_name'] == entry[ch_name_key])[0][0]
                ch_groups_coords = [ch_groups['x'][ch_groups_i], ch_groups['y'][ch_groups_i], ch_groups['z'][ch_groups_i]]
                ## Makes sure the coordinates match (with some lee-way for manual calculations rounding)
                np.testing.assert_allclose(ch_groups_coords, entry[ch_coords_key], atol=0.1,
                                        err_msg=f"The coordinates of channel {entry[ch_name_key]} in the data ({entry[ch_coords_key]}) do not match those in the localisation groups ({ch_groups_coords}).")
                ## Assigns the localisation group
                loc_groups.append(ch_groups['group'][ch_groups_i])
            else:
                if handle_missing == "error":
                    raise ValueError(f"The localisation group for channel {entry[ch_name_key]} is missing from the file.")
                elif handle_missing == "warning":
                    print(f"The localisation group for channel {entry[ch_name_key]} is missing from the file, using 'unassigned' instead.")
                    loc_groups.append('unassigned')

    ### Adds the localisation groups
    data['loc_group'] = loc_groups

    return data