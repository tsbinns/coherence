import sys 
import os
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import matplotlib; matplotlib.use('TKAgg')


""" Gets path info """
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(MAIN_PATH, 'coherence'))
BIDS_PATH = os.path.join(MAIN_PATH, 'data')

from annotations import annotate


""" Setup """
sub = '004'
ses = 'EphysMedOff01'
task = 'Rest'
acq = 'StimOff'
run = '02'
datatype = 'ieeg'


""" Loads data """
bids_path = BIDSPath(subject=sub, session=ses, task=task, acquisition=acq, run=run, root=BIDS_PATH)
ANNOT_PATH = os.path.join(BIDS_PATH, 'sub-'+sub, 'ses-'+ses, datatype, bids_path.basename+'_annotations.csv')
raw = read_raw_bids(bids_path=bids_path, verbose=False)


""" Plots data for annotation """
annotate(raw, ANNOT_PATH)