"""Analyses ECoG-LFP coherence values to produce cohort-wise results."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_power_analysis import power_standard_analysis


### Info about the results to analyse
FOLDERPATH_PROCESSING = (
    "\\\\?\\C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Analysis\\Processing"
)
FOLDERPATH_ANALYSIS = (
    "\\\\?\\C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Analysis\\Analysis"
)
ANALYSIS = "pow_standard_whole-MedOffOn"


if __name__ == "__main__":

    analysed = power_standard_analysis(
        folderpath_processing=FOLDERPATH_PROCESSING,
        folderpath_analysis=FOLDERPATH_ANALYSIS,
        analysis=ANALYSIS,
        save=True,
    )
