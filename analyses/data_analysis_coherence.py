"""Analyses ECoG-LFP coherence values to produce cohort-wise results."""


import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_analysis import coherence_analysis
from coh_handle_files import load_file


### Info about the data to analyse
EXTRAS_FOLDERPATH = "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN"
RESULTS_FOLDERPATH = EXTRAS_FOLDERPATH + "\\BIDS_Berlin_ECOG_LFP"
SETTINGS_FOLDERPATH = EXTRAS_FOLDERPATH + "\\settings"
ANALYSIS = "ECOG-LFP_coherence-results_all.json"
TO_ANALYSE = load_file(SETTINGS_FOLDERPATH + "\\" + ANALYSIS)


if __name__ == "__main__":

    analysed = coherence_analysis(
        results_folderpath=RESULTS_FOLDERPATH,
        to_analyse=TO_ANALYSE,
        result_types=["connectivity-coh", "connectivity-imcoh"],
    )