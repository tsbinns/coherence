import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, 'coherence'))
from coh_preprocess_data import preprocessing



### Info about the data to analyse
folderpath_data = 'D:\\Data'
folderpath_extras = 'C:\\Users\\tomth\\OneDrive\\My Documents\\Data\\ICN'
dataset = 'BIDS_Berlin_ECOG_LFP'
analysis = 'ECOG-LFP_coherence'
subject = '001'
session = 'EphysMedOff01'
task = 'Rest'
acquisition = 'StimOff'
run = '01'



if __name__ == "__main__":

    preprocessed = preprocessing(
        folderpath_data, folderpath_extras, dataset, analysis, subject, session,
        task, acquisition, run
    )

    print("my name is jeff")


