import sys 
import os
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import matplotlib; matplotlib.use('TKAgg')


""" Gets path info """
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(MAIN_PATH, 'coherence'))
BIDS_PATH = os.path.join(MAIN_PATH, 'data')

import preprocessing
import processing
import plotting


""" Setup """
# Data
sub = '004'
ses = 'EphysMedOff01'
task = 'Rest'
acq = 'StimOff'
run = '01'
datatype = 'ieeg'

# Analysis
# Preprocessing settings
chans = ['ECOG_L_1_SMC_AT', 'ECOG_L_2_SMC_AT', 'ECOG_L_3_SMC_AT',
         'ECOG_L_4_SMC_AT', 'ECOG_L_5_SMC_AT', 'ECOG_L_6_SMC_AT',
         'LFP_L_1_STN_BS', 'LFP_L_8_STN_BS']
resample = 250 # Hz
highpass = 3 # Hz
lowpass = 125 # Hz
notch = np.arange(50, lowpass+1, 50) # Hz
epoch_len = 5 # seconds
inc_shuffled = True

# Spectral analysis settings
l_freq = highpass
h_freq = 100
normalise_psd = True
cwt_freqs = np.arange(highpass, h_freq+1)
coh_method = 'imcoh'


""" Loads data """
bids_path = BIDSPath(subject=sub, session=ses, task=task, acquisition=acq,
                     run=run, root=BIDS_PATH)
ANNOT_PATH = os.path.join(BIDS_PATH, 'sub-'+sub, 'ses-'+ses, datatype,
                          bids_path.basename+'_annotations.csv')
raw = read_raw_bids(bids_path=bids_path, verbose=False)
annots = None #read_annotations(ANNOT_PATH)


""" Analysis """
processed = preprocessing.process(raw, annotations=annots, channels=chans, resample=resample,
                                  highpass=highpass, lowpass=lowpass, notch=notch,
                                  epoch_len=epoch_len, include_shuffled=inc_shuffled)
psds = processing.get_psd(processed, l_freq, h_freq, normalise_psd, notch[0])
cohs = processing.get_coherence(processed, cwt_freqs, coh_method)


""" Plotting """
plotting.psd(psds, freq_limit=50, normalise=normalise_psd)
plotting.coherence(cohs, freq_limit=50)