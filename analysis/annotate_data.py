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

from annotations import annotate
from helpers import filter_for_annotation

# Setup
sub = 'FOGC001'
ses = 'EphysMedOff01'
task = 'Rest'
acq = 'StimOff'
run = '01'
datatype = 'ieeg'

channels = ["ECOG_R_1_SM_HH", "ECOG_R_2_SM_HH", "ECOG_R_3_SM_HH", "ECOG_R_4_SM_HH",
                                             "ECOG_R_5_SM_HH", "ECOG_R_6_SM_HH", "ECOG_R_7_SM_HH", "ECOG_R_8_SM_HH",
                                             "LFP_R_1_STN_PI", "LFP_R_4_STN_PI"]
filter_data = True

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
annotate(raw, annot_path)