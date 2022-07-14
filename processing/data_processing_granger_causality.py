"""Processes ECoG and LFP data to generate Granger causality values."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_processing import granger_processing
from coh_loading import load_preprocessed_dict


### Info about the data to analyse
FOLDERPATH_PREPROCESSING = (
    "\\\\?\\C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Analysis\\Preprocessing"
)
FOLDERPATH_PROCESSING = (
    "\\\\?\\C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Analysis\\Processing"
)
DATASET = "BIDS_Berlin_ECOG_LFP"
PREPROCESSING = "preprocessed-with_notch_filter"
ANALYSIS = "con_granger_whole"
SUBJECT = "006"
SESSION = "EcogLfpMedOff02"
TASK = "Rest"
ACQUISITION = "StimOff"
RUN = "1"


if __name__ == "__main__":

    preprocessed = load_preprocessed_dict(
        folderpath_preprocessing=FOLDERPATH_PREPROCESSING,
        dataset=DATASET,
        preprocessing=PREPROCESSING,
        subject=SUBJECT,
        session=SESSION,
        task=TASK,
        acquisition=ACQUISITION,
        run=RUN,
    )

    granger_processing(
        preprocessed,
        FOLDERPATH_PROCESSING,
        DATASET,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        save=True,
    )
