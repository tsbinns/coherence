"""Analyses ECoG-LFP coherence values."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_analysis import connectivity_analysis


### Info about the results to analyse
FOLDERPATH_ANALYSIS = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Analysis"
)
FOLDERPATH_PROCESSING = (
    "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN\\Processing"
)
ANALYSIS = "ECOG-LFP_coherence-MedOffOn_connectivity"


if __name__ == "__main__":

    analysed = connectivity_analysis(
        folderpath_processing=FOLDERPATH_PROCESSING,
        folderpath_analysis=FOLDERPATH_ANALYSIS,
        analysis=ANALYSIS,
        save=True,
    )
