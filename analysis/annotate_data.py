import sys 
import os
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import matplotlib; matplotlib.use('TKAgg')


# Path info
cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))

main_path = 'C:\\Users\\tomth\\OneDrive\\Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata'
project_path = 'C:\\Users\\tomth\\OneDrive\\Documents\\Work\\Courses\\Berlin\\ECN\\ICN\\Data\\BIDS_Berlin_ECOG_LFP\\rawdata\\projects\\coherence'

from annotations import annotate

# Setup
sub = '004'
ses = 'EphysMedOff01'
task = 'Rest'
acq = 'StimOff'
run = '02'
datatype = 'ieeg'

# Loads data
bids_path = BIDSPath(subject=sub, session=ses, task=task, acquisition=acq, run=run, root=main_path)
annot_path = os.path.join(project_path, 'annotations', bids_path.basename+'.csv')
raw = read_raw_bids(bids_path=bids_path, verbose=False)

# Plots data for annotation
annotate(raw, annot_path)