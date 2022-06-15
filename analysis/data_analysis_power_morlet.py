"""Analyses ECoG-LFP coherence values to produce cohort-wise results."""


import os
import sys
from collections import OrderedDict
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_power_analysis import morlet_analysis
from coh_handle_files import load_file


### Info about the results to analyse
EXTRAS_FOLDERPATH = "C:\\Users\\User\\OneDrive\\My Documents\\Data\\ICN"
RESULTS_FOLDERPATH = EXTRAS_FOLDERPATH + "\\BIDS_Berlin_ECOG_LFP"
SETTINGS_FOLDERPATH = EXTRAS_FOLDERPATH + "\\settings"
ANALYSIS = "ECOG-LFP_coherence-results_all.json"
TO_ANALYSE = load_file(SETTINGS_FOLDERPATH + "\\" + ANALYSIS)

### How to analyse the results
ANALYSIS_STEPS = OrderedDict([("average", "runs"), ("average", "sessions")])


if __name__ == "__main__":

    analysed = morlet_analysis(
        results_folderpath=RESULTS_FOLDERPATH,
        to_analyse=TO_ANALYSE,
        analysis_steps=ANALYSIS_STEPS,
    )
