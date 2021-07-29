import sys 
import os
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import numpy as np


""" Gets path info """
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(MAIN_PATH, 'coherence'))
BIDS_PATH = os.path.join(MAIN_PATH, 'data')

import preprocessing
import processing


""" Setup """
# Data
sub = '004'
ses = 'EphysMedOff01'
task = 'Rest'
acq = 'StimOff'
run = '01'
datatype = 'ieeg'

# Analysis
chans = ['ECOG_L_1_SMC_AT', 'ECOG_L_2_SMC_AT', 'ECOG_L_3_SMC_AT',
         'ECOG_L_4_SMC_AT', 'ECOG_L_5_SMC_AT', 'ECOG_L_6_SMC_AT',
         'LFP_L_1_STN_BS', 'LFP_L_8_STN_BS']
resample = 250 # Hz
bandpass = [3, 125] # Hz
notch = np.arange(50, bandpass[1]+1, 50) # Hz
epoch_len = 2 # seconds
inc_shuffled = True


""" Loads data """
bids_path = BIDSPath(subject=sub, session=ses, task=task, acquisition=acq,
                     run=run, root=BIDS_PATH)
ANNOT_PATH = os.path.join(BIDS_PATH, 'sub-'+sub, 'ses-'+ses, datatype,
                          bids_path.basename+'_annotations.csv')
raw = read_raw_bids(bids_path=bids_path, verbose=False)
annots = None #read_annotations(ANNOT_PATH)


""" Analysis """
processed = preprocessing.process(raw, annotations=annots, channels=chans,
                                  resample=resample, bandpass=bandpass,
                                  notch=notch, epoch_len=epoch_len,
                                  include_shuffled=inc_shuffled)
psds = processing.get_psd(processed)
cohs = processing.get_coherence(processed)
