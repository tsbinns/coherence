import matplotlib.pyplot as plt
import os.path


def annotate(raw, annot_path, orig_time=[]):
    """ Plot raw data for annotation.

    PARAMETERS
    ----------
    raw : MNE Raw object
    -   The data to be annotated.

    annot_path : str
    -   Filepath to where the annotations should be saved.

    orig_time : list (default), None, or datetime
    -   What 'orig_time' to assign the annotations. If '[]' (default), the time of raw data acquisition is used. If
        None, no time is assigned. If a datetime, that timestamp is used.


    RETURNS
    ----------
    N/A
    
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

    # Sets the annotation 'orig_time'
    if orig_time != []:
        raw.annotations.orig_time = orig_time

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

