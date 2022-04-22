"""Processes ECoG and LFP data to generate power spectra."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_power_processing import fooof_analysis, morlet_analysis
from coh_preprocess_data import preprocessing


### Info about the data to analyse
FOLDERPATH_DATA = (
    "C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\PROJECT "
    "ECOG-LFP Coherence\\Data"
)
FOLDERPATH_PROCESSING = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Processing"
)
DATASET = "BIDS_Berlin_ECOG_LFP"
ANALYSIS = "ECOG-LFP_coherence-power_fooof_processing"
SUBJECT = "001"
SESSION = "EcogLfpMedOn02"
TASK = "Rest"
ACQUISITION = "StimOff"
RUN = "1"


if __name__ == "__main__":

    preprocessed = preprocessing(
        FOLDERPATH_DATA,
        FOLDERPATH_PROCESSING,
        DATASET,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
    )

    power = morlet_analysis(
        preprocessed,
        FOLDERPATH_PROCESSING,
        DATASET,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        save=False,
    )

    fooof_analysis(
        power,
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
