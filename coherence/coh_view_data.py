"""Plots data for annotating.

METHODS
-------
annotate_data
-   Plots non-epoched signals for annotating.
"""

from os.path import exists
from coh_handle_files import generate_sessionwise_fpath
from coh_signal_viewer import SignalViewer
import coh_signal


def annotate_data(
    signal: coh_signal.Signal,
    folderpath_preprocessing: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
) -> None:
    """
    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to plot.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing folder.

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    subject : str
    -   The name of the subject whose data will be plotted.

    session : str
    -   The name of the session for which the data will be plotted.

    task : str
    -   The name of the task for which the data will be plotted.

    acquisition : str
    -   The name of the acquisition mode for which the data will be plotted.

    run : str
    -   The name of the run for which the data will be plotted.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    annotations_fpath = generate_sessionwise_fpath(
        f"{folderpath_preprocessing}\\Settings\\Specific",
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "annotations",
        ".csv",
    )

    ### Data plotting
    ## Plots the data for annotating
    signal_viewer = SignalViewer(signal=signal)
    if exists(annotations_fpath):
        signal_viewer.load_annotations(fpath=annotations_fpath)
    signal_viewer.plot()
    signal_viewer.save_annotations(fpath=annotations_fpath)
