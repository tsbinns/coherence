import matplotlib.pyplot as plt
import os.path
from mne import events_from_annotations, annotations_from_events


def annotate(raw, annot_path):
    """ Plot raw data for annotation.

    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data to be annotated.

    annot_path : str
    -   Filepath to where the annotations should be saved.


    RETURNS
    ----------
    raw : MNE Raw object
    -   The data with the annotations.
    
    """
    
    # Sets LFP channel types to type DBS
    names = raw.info.ch_names
    new_types = {}
    for name in names:
        if name[:3] == 'LFP':
            new_types[name] = 'dbs'
    raw.set_channel_types(new_types)

    # Shows the data for annotation
    raw.plot(scalings='auto')
    plt.tight_layout()

    # Saves the annotations
    get_answer = False
    while not get_answer:
        save = input('Do you want to save these annotations? (y/n): ')
        if save == 'y':
            if os.path.isfile(annot_path):
                save = input('This will overwite the existing annotations. Do you want to continue? (y/n): ')
                if save == 'y':
                    print('Saving annotations to: ', annot_path)
                    raw.annotations.save(annot_path, overwrite=True)
                    get_answer = True
                else:
                    break
            else:
                print('Saving annotations to: ', annot_path)
                raw.annotations.save(annot_path, overwrite=True)
                get_answer = True
        elif save == 'n':
            print('Discarding annotations')
            get_answer = True
        else:
            print('This is not a valid input. Try again.')


    return raw



def set_orig_time(raw, annot_path, orig_time):
    """ Sets the orig_time parameter for a set of annotations.

    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The Raw object with annotations whose orig_time will be modified.

    annot_path : str
    -   Filepath to where the annotations should be saved.

    orig_time : None, DateTime
    -   The new orig_time.

    RETURNS
    ----------
    raw : MNE Raw object
    -   The Raw object with annotations with the new orig_time.
    
    """

    ### Extracts events from the annotations
    events = events_from_annotations(raw, event_id=None)

    ### Creates new annotations from the events with the new orig_time
    annots = annotations_from_events(events=events, sfreq=raw.sfreq, orig_time=orig_time)

    ### Adds the modified annotations to the raw object and saves them
    ## Adds annotations
    raw.set_annotations(annots)
    ## Saves them
    get_answer = False
    while not get_answer:
        save = input(f"Do you want to set the annotation start times to {orig_time}? (y/n): ")
        if save == 'y':
            print('Saving annotations with the specified orig_time to: ', annot_path)
            raw.annotations.save(annot_path, overwrite=True)
            get_answer = True
        elif save == 'n':
            print('Discarding annotations.')
            get_answer = True
        else:
            print('This is not a valid input. Try again.')


    return raw