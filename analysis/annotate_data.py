import sys 
import os
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import matplotlib; matplotlib.use('TKAgg')


# Path info
cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))

main_path = 'C:\\Users\\tomth\\Data\\BIDS_Beijing_ECOG_LFP\\rawdata'
project_path = 'C:\\Users\\tomth\\OneDrive\\My Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Beijing_ECOG_LFP\\projects\\coherence'

from annotations import annotate, set_orig_time
from helpers import filter_for_annotation

# Setup
sub = 'FOG011'
ses = 'EphysMedOff01'
task = 'Rest'
acq = 'StimOff'
run = '01'
datatype = 'ieeg'

channels = ["ECOG_R_1_SM_HH", "ECOG_R_2_SM_HH", "ECOG_R_3_SM_HH", "ECOG_R_4_SM_HH",
                                             "ECOG_R_5_SM_HH", "ECOG_R_6_SM_HH", "ECOG_R_7_SM_HH", "ECOG_R_8_SM_HH",
                                             "ECOG_R_9_SM_HH", "ECOG_R_10_SM_HH", "ECOG_R_11_SM_HH", "ECOG_R_12_SM_HH",
                                             "ECOG_R_13_SM_HH", "ECOG_R_14_SM_HH", "ECOG_R_15_SM_HH", "ECOG_R_16_SM_HH",
                                             "ECOG_R_17_SM_HH", "ECOG_R_18_SM_HH", "ECOG_R_19_SM_HH", "ECOG_R_20_SM_HH",
                                             "ECOG_R_21_SM_HH", "ECOG_R_22_SM_HH", "ECOG_R_23_SM_HH", "ECOG_R_24_SM_HH",
                                             "ECOG_R_25_SM_HH", "ECOG_R_26_SM_HH", "ECOG_R_27_SM_HH", "ECOG_R_28_SM_HH",
                                             "ECOG_R_29_SM_HH", "ECOG_R_30_SM_HH",
                                             "LFP_R_1_STN_PI", "LFP_R_4_STN_PI"]
filter_data = True
orig_time = []

# Loads data
bids_path = BIDSPath(subject=sub, session=ses, task=task, acquisition=acq, run=run, root=main_path)
annot_path = os.path.join(project_path, 'annotations', bids_path.basename+'.csv')
raw = read_raw_bids(bids_path=bids_path, verbose=False)

if channels != None:
    for channel in channels:
        if channel not in raw.info.ch_names:
            raise ValueError(f'The requested channel {channel} is not present in the data.')
    raw.pick_channels(channels)

if filter_data == True:
    raw = filter_for_annotation(raw)

# Plots data for annotation
raw = annotate(raw, annot_path)

### Alters the orig_time of the annotations, if requested
if orig_time != []:
    raw = set_orig_time(raw, annot_path, orig_time)
    print('jeff')