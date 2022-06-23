"""Plots ECoG-LFP connectivity values."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_plotting import connectivity_plotting


### Info about the results to analyse
FOLDERPATH_ANALYSIS = (
    "\\\\?\\C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Analysis\\Analysis"
)
FOLDERPATH_PLOTTING = (
    "\\\\?\\C:\\Users\\User\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Analysis\\Plotting"
)
PLOTTING = "con_multivar_regional-MedOffOn"


if __name__ == "__main__":

    analysed = connectivity_plotting(
        folderpath_analysis=FOLDERPATH_ANALYSIS,
        folderpath_plotting=FOLDERPATH_PLOTTING,
        plotting=PLOTTING,
        save=True,
    )
