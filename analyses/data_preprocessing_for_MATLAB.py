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
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_handle_files import generate_sessionwise_fpath
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

### Where to save the pre-processed data
SAVE_FPATH = generate_sessionwise_fpath(
    FOLDERPATH_EXTRAS,
    DATASET,
    SUBJECT,
    SESSION,
    TASK,
    ACQUISITION,
    RUN,
    "preprocessed",
    ".json",
)


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
    ).save_data(fpath=SAVE_FPATH)

    print("my name is jeff")
