"""Processes ECoG and LFP data to generate coherence values."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_processing import coherence_processing
from coh_preprocess_data import preprocessing


### Info about the data to analyse
FOLDERPATH_DATA = (
    "C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\PROJECT "
    "ECOG-LFP Coherence\\Data"
)
FOLDERPATH_EXTRAS = "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN"
DATASET = "BIDS_Berlin_ECOG_LFP"
ANALYSIS = "ECOG-LFP_coherence"
SUBJECT = "001"
SESSION = "EcogLfpMedOff01"
TASK = "Rest"
ACQUISITION = "StimOff"
RUN = "1"


if __name__ == "__main__":

    preprocessed = preprocessing(
        FOLDERPATH_DATA,
        FOLDERPATH_EXTRAS,
        DATASET,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
    )

    coherence_processing(
        preprocessed,
        FOLDERPATH_EXTRAS,
        DATASET,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        save=True,
    )
