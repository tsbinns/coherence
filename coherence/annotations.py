import matplotlib.pyplot as plt
import os.path


def annotate(raw, annot_path):
    """ Plot raw data for annotation.

    PARAMETERS
    ----------
    raw : MNE Raw object
        The data to be annotated.
    ANNOT_PATH : str
        Filepath to where the annotations should be saved.

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

