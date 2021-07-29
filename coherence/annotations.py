import matplotlib.pyplot as plt


def annotate(raw, ANNOT_PATH):

    
    """ Analysis """
    # Sets LFP channel types to type SEEG
    names = raw.info.ch_names
    new_types = {}
    for name in names:
        if name[:3] == 'LFP':
            new_types[name] = 'seeg'
    raw.set_channel_types(new_types)

    # Shows the data for annotation
    raw.pick_types(ecog=True, dbs=True)
    raw.plot(scalings='auto')
    plt.tight_layout()

    # Saves the annotations
    save = input('Do you want to save these annotations? (y/n): ')
    if save == 'y':
        print('Saving annotations to: ', ANNOT_PATH)
        raw.annotations.save(ANNOT_PATH, overwrite=True)
    else:
        print('Discarding annotations')