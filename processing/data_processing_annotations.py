"""Allows users to add and view data annotations."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_preprocess_data import preprocessing_for_annotations
from coh_view_data import annotate_data


### Info about the data to analyse
FOLDERPATH_DATA = (
    "C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Data"
)
FOLDERPATH_PREPROCESSING = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Preprocessing"
)
DATASET = "BIDS_Berlin_ECOG_LFP"
ANALYSIS = "for_annotations"
SUBJECT = "009"
SESSION = "EcogLfpMedOn02"
TASK = "Rest"
ACQUISITION = "StimOffDopa65"
RUN = "1"


if __name__ == "__main__":

    preprocessed = preprocessing_for_annotations(
        FOLDERPATH_DATA,
        FOLDERPATH_PREPROCESSING,
        DATASET,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
    )

    annotate_data(
        preprocessed,
        FOLDERPATH_PREPROCESSING,
        DATASET,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        load_annotations=True,
    )
