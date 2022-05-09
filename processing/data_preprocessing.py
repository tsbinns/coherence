"""Preprocesses ECoG and LFP data and saves the data."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_preprocess_data import preprocessing


### Info about the data to analyse
FOLDERPATH_DATA = (
    "C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Data"
)
FOLDERPATH_PREPROCESSING = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Preprocessing"
)
DATASET = "BIDS_Berlin_ECOG_LFP"
ANALYSIS = "with_window_and_notch_filter"
SUBJECT = "001"
SESSION = "EcogLfpMedOn02"
TASK = "Rest"
ACQUISITION = "StimOff"
RUN = "1"


if __name__ == "__main__":

    preprocessing(
        folderpath_data=FOLDERPATH_DATA,
        folderpath_preprocessing=FOLDERPATH_PREPROCESSING,
        dataset=DATASET,
        analysis=ANALYSIS,
        subject=SUBJECT,
        session=SESSION,
        task=TASK,
        acquisition=ACQUISITION,
        run=RUN,
        save=True,
    )
