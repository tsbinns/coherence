"""Processes ECoG and LFP data to generate power spectra."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_power_processing import fooof_analysis, morlet_analysis
from coh_loading import load_preprocessed_dict


### Info about the data to analyse
FOLDERPATH_PREPROCESSING = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Preprocessing"
)
FOLDERPATH_PROCESSING = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Processing"
)
DATASET = "BIDS_Berlin_ECOG_LFP"
PREPROCESSING = "preprocessed-FOOOF_Moritz"
POWER_ANALYSIS = "pow_morlet"
FOOOF_ANALYSIS = "pow_fooof"
SUBJECT = "001"
SESSION = "EcogLfpMedOff01"
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

    power = morlet_analysis(
        preprocessed,
        FOLDERPATH_PROCESSING,
        DATASET,
        POWER_ANALYSIS,
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
        FOOOF_ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        save=True,
    )
