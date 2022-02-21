"""Processes ECoG and LFP data to generate power and coherence spectra.

So far implemented:
-   Data preprocessing

To be implemented:
-   Power and coherence calculations.
"""




import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))
from coh_power_analysis import power_analysis
from coh_preprocess_data import preprocessing




### Info about the data to analyse
FOLDERPATH_DATA = 'D:\\Data'
FOLDERPATH_EXTRAS = 'C:\\Users\\tomth\\OneDrive\\My Documents\\Data\\ICN'
DATASET = 'BIDS_Berlin_ECOG_LFP'
ANALYSIS = 'ECOG-LFP_coherence'
SUBJECT = '001'
SESSION = 'EphysMedOff01'
TASK = 'Rest'
ACQUISITION = 'StimOff'
RUN = '01'



if __name__ == "__main__":

    preprocessed = preprocessing(
        FOLDERPATH_DATA, FOLDERPATH_EXTRAS, DATASET, ANALYSIS, SUBJECT, SESSION,
        TASK, ACQUISITION, RUN
    )

    power_analysis(
        preprocessed, FOLDERPATH_DATA, FOLDERPATH_EXTRAS, DATASET, ANALYSIS,
        SUBJECT, SESSION, TASK, ACQUISITION, RUN
    )


    print("my name is jeff")
