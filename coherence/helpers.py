import numpy as np
import pandas as pd
from copy import deepcopy



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
    data.notch_filter(notch)

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
            data[i].drop(columns=['index'], inplace=True)

    # Merges data of different types (e.g. data from different runs)
    all_data = pd.concat(data[:], ignore_index=True)


    return all_data



def combine_channel_names(data, ch_keys, joining='&'):
    """ Concatenates channel names together, useful in the event that unique combinations of channels need to be found.
    
    PARAMETERS
    ----------
    data : pandas DataFrame
    -   A DataFrame containing the names of the channels.

    ch_keys : list of strs
    -   The keys of the channel name columns in the DataFrame.

    joining : str
    -   A string with which to join multiple channel names.


    RETURNS
    ----------
    names : list of strs
    -   The list of combined channel names.
    """

    names = [] # get names of all channels
    for chann_i in range(np.shape(data)[0]):
        cat_name = [] # if multiple channel names, combine them 
        for ch_key in ch_keys:
            cat_name.append(data[ch_key][chann_i])
        cat_name = joining.join(cat_name)
        names.append(cat_name)


    return names



def unique_channel_names(ch_names):
    """ Finds the unique channel names.

    PARAMETERS
    ----------
    names : list of strs
    -   Channel names to find the unique entries of.
    

    RETURNS
    ----------
    unique_names : list of strs
    -   Unique channel names.


    unique_idc : list of ints
    -   Indices of each unique channel in names.
    """

    unique_ch_names = pd.unique(ch_names) # find unique channel names
    unique_ch_idc = [] # find indices of these unique channels
    for unique_ch_name in unique_ch_names:
        unique_ch_idc.append([i for i, x in enumerate(ch_names) if x == unique_ch_name])


    return unique_ch_names, unique_ch_idc



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
    data : pandas DataFrame
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

    # Flattens data into a 1D list
    flat_data = [item for sublist in data.values.flatten() for item in sublist]

    # Finds the minimum and maximum values in the dataset
    lim = [0, 0]
    if floor == None:
        lim[0] = min(flat_data)
    else:
        lim[0] = floor
    lim[1] = max(flat_data)

    # If requested, adds a buffer to the axis limits
    if border:
        data_range = lim[1] - lim[0]
        border_val = data_range*(border/100)
        lim[0] -= border_val
        lim[1] += border_val


    return lim



def window_title(info, base_title='', full_info=True):
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
    if full_info == False: # If partial information is requested, provide a general 'avg' tag rather than a tag...
    #... containing details of each piece of data involved
        for key in info.keys():
            for val_i, val in enumerate(info[key]):
                if val[:3] == 'avg': # if the data has been averaged, discard any additional info
                    info[key][val_i] = 'avg'
            info[key] = pd.unique(info[key])

    ## Title creation
    title = base_title
    included = []
    first = True
    for key in info.keys():
        if len(info[key]) == 1: # if only a single type of data is present, add this information to the window's title
            if first == True: # if this is the first key to be added, don't add a comma to the start
                title += f'{key}-{info[key][0]}'
                first = False
            else: # if this is not the first key to be added, add a comma to separate the different factors
                title += f',{key}-{info[key][0]}'
            included.append(key)


    return title, included



def channel_title(info, already_included=[], base_title='', full_info=True):
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

    ## Title creation
    title = base_title
    included = []
    first = True
    for key in info.keys():
        if len(info[key]) == 1 and key not in already_included: # if only a single type of data is present and this...
        #... information has not been included in the window's title, add this information to the channel's title
            if len(title) >= 40: # adds a new line to the title to stop it getting too wide
                title += '\n'
            if first == True: # if this is the first key to be added, don't add a comma to the start
                title += f'{key}-{info[key][0]}'
                first = False
            else: # if this is not the first key to be added, add a comma to separate the different factors
                title += f',{key}-{info[key][0]}'
            included.append(key)


    return title, included



def data_title(info, already_included=[], full_info=True):
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

    ## Title creation
    title = ''
    first = True
    for key in info.keys():
        if key not in already_included:  # if only a single type of data is present and this...
        #... information has not been included in the window's and subplot's titles, add this information to the data...
        #... label
            if first == True:  # if this is the first key to be added, don't add a comma to the start
                title += f'{key}-{info[key][0]}'
                first = False
            else: # if this is not the first key to be added, add a comma to separate the different factors
                title += f',{key}-{info[key][0]}'


    return title



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
            elif info[key][0] == 'non-binary': # if many types of data can be present, randomly generate colours
                for x in info[key][1]:
                    colours[key].append(np.random.rand(1, 3))
            else:
                raise ValueError(f'The colour key {info[key][0]} is not recognised. Only binary and non-binary are accepted.')


    return colours



def check_identical(data, return_result=False):
    """ Checks to see if multiple sets of data are identical.

    PARAMETERS
    ----------
    data : array
    -   A list containing the data to be compared.

    return_result : bool, default False
    -   Whether or not to raise an error if the data is not identical. If False (default), an error is raised and the
        script stops. If True, the script is not stopped, but the non-identical nature of the data is returned.
    

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
                if all(sample_1) != all(sample_2): #... check if the values are equal
                    if return_result == True:
                        identical = False
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
    """

    # Converts data to a workable type
    if type(data) != list:
        data = list(data)

    # Processes data
    avg = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)


    return avg, std



def average_dataset(data, avg_over, ch_keys, x_keys, y_keys):
    """ Averages data over specific factors (e.g. runs, subjects).

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data to average.

    avg_over : str
    -   The column name of the data to average over.

    ch_keys : list of strs
    -   The names of the columns in the DataFrame containing the channel names, so that the correct data entries can be
        averaged over.

    x_keys : list of strs
    -   The names of the columns in the DataFrame containing the keys whose values should be identical for the entries
        that are being averaged (i.e. the independent/control variables).

    y_keys : list of strs
    -   The names of the columns in the DataFrame whose values should be averaged.


    RETURNS
    ----------
    new_data : pandas DataFrame
    -   The averaged data.
    """

    ### Setup
    condition_keys = ['med', 'stim', 'task', 'subject', 'run']

    # Makes sure you won't average data over different conditions
    for cond_key in condition_keys:
        if cond_key != avg_over:
            if len(np.unique(data[cond_key])) > 1:
                raise ValueError(f'You are trying to average data over {avg_over}, but you are also averaging data over {cond_key}.')

    # Finds channels to average over
    ch_names = combine_channel_names(data, ch_keys)
    _, unique_idc = unique_channel_names(ch_names)

    # Makes sure the x_keys each unique channel are identical
    for x_key in x_keys: # for each key whose values should be identical
        for unique_i in unique_idc: # for each unique channel
            identical = check_identical(list(data[x_key][unique_i])) # the values to compare
            if identical == False:
                raise ValueError(f'The {x_key} values for the data do not match, and therefore the data cannot be averaged over.')
    

    ### Processing
    # Averages across the y_keys for each unique channel
    all_avg = []
    all_std = []
    for y_key in y_keys:
        all_avg.append([])
        all_std.append([])
        for unique_i in unique_idc:
            avg, std = average_data(data[y_key][unique_i])
            all_avg[-1].append(avg)
            all_std[-1].append(std)
    avg = all_avg
    std = all_std

    # Makes new names for the averaged-over factor
    new_name = []
    for unique_i in unique_idc:
        new_name.append(f'avg{list(data[avg_over][unique_i])}')
        #new_name.append('avg')

    # Adds new columns for std data of each y_key (if not already present)
    holder = np.repeat(np.nan, np.shape(data)[0])
    for y_key in y_keys:
        if y_key+'_std' not in data.columns:
            y_key_pos = list(data.columns).index(y_key)
            data.insert(y_key_pos+1, y_key+'_std', holder)

    # Collates data for new DataFrame
    new_cols = []
    avg_i = 0
    std_i = 0
    for key in data.columns:
        new_cols.append([])
        if key == avg_over: # the averaged-over factor gets a special name, specifying what was averaged over
        #... (e.g. run IDs, subject IDs, etc...)
            for unique_i in unique_idc:
                new_cols[-1].append(f'avg{list(data[key][unique_i])}')
        elif key in y_keys: # the values that were averaged get used in place of the original values
            new_cols[-1].extend(avg[avg_i])
            avg_i += 1
        elif 'std' in key: # the std values (should not be in y_keys, as you don't want to average over std)
            new_cols[-1].extend(std[std_i])
            std_i += 1
        else: # the non-averaged/non-averaged-over entries can be repeated from the original data
            for unique_i in unique_idc:
                new_cols[-1].append(data[key][unique_i[0]])
    
    # Creates new DataFrame
    new_data = list(zip(*new_cols[:]))
    new_data = pd.DataFrame(data=new_data, columns=list(data.columns))
            

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



def average_shuffled(data, keys_to_avg, channel_name_key='ch_name'):
    """ Averages data over adjacent shuffled channels of the same type.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The data to average over. Must contain a column that can be used to identify the shuffled channels.

    keys_to_avg : list of str
    -   The keys of the DataFrame columns to be averaged.

    channel_name_key : str
    -   The name of the column used to identify the shuffled channels. 'ch_name' by default.


    RETURNS
    ----------
    data : pandas DataFrame
    -   The data with entries from adjacent shuffled channels of the same type averaged over.
    """

    # Find the unique shuffled channels based on their positions
    i = 0
    avg_over = []
    for idx, name in enumerate(data[channel_name_key]): # for each channel
        if name[:9] != 'SHUFFLED-': # if it's not a shuffled channel, move on
            avg_over.append([])
            i += 1
        else:
            if name[:10] == 'SHUFFLED-0': # if it's the start of a shuffled channel run, add it to the index
                avg_over.append([])
                avg_over[i] = [idx]
            else: # and add any subsequent channels
                avg_over[i].extend([idx])

    # Excludes the empty entries from avg_over
    empty = []
    for i in range(len(avg_over)):
        if not avg_over[i]:
            empty.append(i)
    avg_over = np.delete(avg_over, empty, 0).tolist()

    # Averages the shuffled data across the unique channels
    discard = [] # entries to remove after averaging
    for over in avg_over:
        if len(over) > 1: # if there is more than one piece of data
            for key in keys_to_avg: # average the data
                data[key][over[0]] = np.mean(data[key][over].tolist(), axis=0)
                # SHOULD ADD CHECK TO MAKE SURE DATA IS 1D
            discard.extend(over[1:]) # adds entries to remove later
        # renames the remaining channel
        name = data[channel_name_key][over[0]]
        data[channel_name_key][over[0]] = name[:name.index('-')] + name[name.index('_'):]
    data.drop(discard, inplace=True) # deletes the redundant entries
    data.reset_index(inplace=True)


    return data



def coherence_by_band(data, methods, band_names=None):
    """ Calculates band-wise coherence values based on the frequency-wise data.

    PARAMETERS
    ----------
    data : pandas DataFrame
    -   The frequency-wise coherence data.

    methods : list of str
    -   The methods used to calculate coherence.

    band_names : list of str | None (default)
    -   The frequency bands to calculate band-wise coherence values for. If None (default), all frequency bands provided
        by the helpers.freq_band_info function are used.
    
    
    RETURNS
    ----------
    data : pandas DataFrame
    -   The original frequency-wise coherence data, plus band-wise coherence values. These values include: the average
        coherence in each band (avg); the maximum coherence in each band (max); and the frequency at which this maximum
        coherence occurs (fmax).
    """

    # Gets the band-wise data
    bands = freq_band_info(band_names) # the frequency bands to analyse

    band_data = {}
    band_data['bands'] = []
    band_data['avg'] = [] # average coherence in each frequency band
    band_data['max'] = [] # maximum coherence in each frequency band
    band_data['fmax'] = [] # frequency of maximum coherence in each frequency band
    for method_i, method in enumerate(methods): # for each coherence calculation method
        band_data['avg'].append([])
        band_data['max'].append([])
        band_data['fmax'].append([])

        for i in range(len(data[method])): # finds the band-wise coherence for each channel
            band_is = freq_band_indices(data['freqs'][i], bands)
            if method_i == 0:
                band_data['bands'].append(list(bands.keys()))
            band_data['avg'][method_i].append([])
            band_data['max'][method_i].append([])
            band_data['fmax'][method_i].append([])
            for key in bands.keys():
                band_data['avg'][method_i][i].append(data[method][i][band_is[key][0]:band_is[key][1]+1].mean())
                band_data['max'][method_i][i].append(data[method][i][band_is[key][0]:band_is[key][1]+1].max())
                band_data['fmax'][method_i][i].append(data['freqs'][i][int(np.where(data[method][i] == 
                                                      band_data['max'][method_i][i][-1])[0])])
    
    # Collects band-wise data
    band_data = list(zip(band_data['bands'], *band_data['avg'][:], *band_data['max'][:], *band_data['fmax'][:]))
    fbands_keynames = ['_fbands_avg', '_fbands_max', '_fbands_fmax']
    fbands_keys = []
    for keyname in fbands_keynames:
        for method in methods:
            fbands_keys.append(method+keyname)
    band_data = pd.DataFrame(data=band_data, columns=['fbands', *fbands_keys])

    # Collects frequency- and band-wise data
    data = pd.concat([data, band_data], axis=1)


    return data