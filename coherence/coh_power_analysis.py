"""Performs power analysis on pre-processed data.

METHODS
-------
power_analysis
-   Takes a coh_signal.Signal object of pre-processed data and performs Morlet
    wavelet power analysis and FOOOF power analysis.
"""




from coh_filepath import SessionwiseFilepath
from coh_power import PowerMorlet
import coh_signal




def power_analysis(
    signal: coh_signal.Signal,
    folderpath_extras: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str
    ) -> None:
    """
    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_data : str
    -   The folderpath to the location of the datasets.

    folderpath_extras : str
    -   The folderpath to the location of the datasets' 'extras', e.g. the
        annotations, processing settings, etc...

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    subject : str
    -   The name of the subject whose data will be analysed.

    session : str
    -   The name of the session for which the data will be analysed.

    task : str
    -   The name of the task for which the data will be analysed.

    acquisition : str
    -   The name of the acquisition mode for which the data will be analysed.

    run : str
    -   The name of the run for which the data will be analysed.
    """

    morlet_fpath = SessionwiseFilepath(
        folderpath_extras, dataset, subject, session, task, acquisition, run,
        'power-morlet', ''
    ).path()

    morlet = PowerMorlet(signal)
    morlet.process()
    morlet.save(morlet_fpath)

    fooof_fpath = SessionwiseFilepath(
        folderpath_extras, dataset, subject, session, task, acquisition, run,
        'power-fooof', ''
    ).path()

    fooof = PowerFOOOF(signal)
    fooof.process()
    fooof.save(fooof_fpath)
