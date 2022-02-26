"""Generates filepaths in the MNE-BIDS format.

CLASSES
-------
Filepath : abstract base class
-   Abstract class for generating filepaths.

RawFilepath : subclass of the abstract base class Filepath
-   Generates an mne_bids.BIDSPath object for loading an mne.io.Raw object.

SessionwiseFilepath : subclass of the abstract base class Filepath
-   Generates a filepath for an object that corresponds to an individual
    recording session based on the MNE data-storage filepath structure.

AnalysiswiseFilepath : subclass of the abstract base class Filepath
-   Generates a filepath for an object that corresponds to a particular
    analysis spanning multiple recordings sessions.
"""




import os
from abc import ABC, abstractmethod
import mne_bids




class Filepath(ABC):
    """Abstract class for generating filepaths.

    METHODS
    -------
    path (abstract)
    -   Generates the filepath.

    SUBCLASSES
    ----------
    RawFilepath
    -   Generates an mne_bids.BIDSPath object for loading an mne.io.Raw object.

    SessionwiseFilepath
    -   Generates a filepath for an object that corresponds to an individual
        recording session based on the MNE data-storage filepath structure.

    AnalysiswiseFilepath
    -   Generates a filepath for an object that corresponds to a particular
        analysis spanning multiple recordings sessions.
    """

    @abstractmethod
    def path(self):
        """Generates the filepath."""



class RawFilepath(Filepath):
    """Generates an mne_bids.BIDSPath object for loading an mne.io.Raw object.

    PARAMETERS
    ----------
    folderpath : str
    -   The path of the folder where the datasets are located.

    dataset : str
    -   The name of the dataset within the folder given in 'folderpath'.

    subject : str
    -   The name of the subject for which the mne_bids.BIDSPath object should be
        generated.

    session : str
    -   The name of the session for which the mne_bids.BIDSPath object should be
        generated.

    task : str
    -   The name of the task for which the mne_bids.BIDSPath object should be
        generated.

    acquisition : str
    -   The name of the acquisition mode for which the mne_bids.BIDSPath object
        should be generated.

    run : str
    -   The name of the run for which the mne_bids.BIDSPath object should be
        generated.

    METHODS
    -------
    path
    -   Generates the mne_bids.BIDSPath object.
    """

    def __init__(self,
        folderpath: str,
        dataset: str,
        subject: str,
        session: str,
        task: str,
        acquisition: str,
        run: str
        ) -> None:

        self.folderpath = folderpath
        self.dataset = dataset
        self.subject = subject
        self.session = session
        self.task = task
        self.acquisition = acquisition
        self.run = run


    def path(self) -> mne_bids.BIDSPath:
        """Generates an mne_bids.BIDSPath object for loading an mne.io.Raw
        object.

        RETURNS
        -------
        mne_bids.BIDSPath
        -   An mne_bids.BIDSPath object for loading an mne.io.Raw object.
        """

        return mne_bids.BIDSPath(
            subject=self.subject, session=self.session, task=self.task,
            acquisition=self.acquisition, run=self.run,
            root=os.path.join(self.folderpath, self.dataset, 'rawdata')
        )



class SessionwiseFilepath(Filepath):
    """Generates a filepath for an object that corresponds to an individual
    recording session based on the MNE data-storage filepath structure.

    PARAMETERS
    ----------
    folderpath : str
    -   The path of the folder where the datasets are located.

    dataset : str
    -   The name of the dataset folder within the folder given in 'folderpath'.

    subject : str
    -   The name of the subject for which the filepath should be generated.

    session : str
    -   The name of the session for which the filepath should be generated.

    task : str
    -   The name of the task for which the filepath should be generated.

    acquisition : str
    -   The name of the acquisition mode for which the filepath should be
        generated.

    run : str
    -   The name of the run for which the filepath should be generated.

    group_type : str
    -   The name of the group of files for which the filepath should be
        generate, e.g. 'annotations', 'settings'.

    filetype : str
    -   The file extension, prefixed with a period, e.g. '.json', '.csv'.

    METHODS
    -------
    path
    -   Generates the filepath.
    """

    def __init__(self,
        folderpath: str,
        dataset: str,
        subject: str,
        session: str,
        task: str,
        acquisition: str,
        run: str,
        group_type: str,
        filetype: str
        ) -> None:

        self.folderpath = folderpath
        self.dataset = dataset
        self.subject = subject
        self.session = session
        self.task = task
        self.acquisition = acquisition
        self.run = run
        self.group_type = group_type
        self.filetype = filetype

        self._subfolders()
        self._filename()


    def _subfolders(self) -> str:
        """Generates the path for the subfolders within 'folderpath'."""

        self.subfolders = (
            f"{self.dataset}\\sub-{self.subject}\\ses-{self.session}"
        )


    def _filename(self) -> None:
        """Generates the name of the file located within 'folderpath' and
        subfolders path.
        """

        self.filename = (
            f"sub-{self.subject}_ses-{self.session}_task-{self.task}_"
            f"acq-{self.acquisition}_run-{self.run}_{self.group_type}"
            f"{self.filetype}"
        )


    def path(self) -> str:
        """Generates the filepath for an object that corresponds to an
        individual recording session based on the MNE data-storage filepath
        structure.

        RETURNS
        -------
        str
        -   The filepath of the object.
        """

        return os.path.join(self.folderpath, self.subfolders, self.filename)



class AnalysiswiseFilepath(Filepath):
    """Generates a filepath for an object that corresponds to a particular
    analysis spanning multiple recordings sessions.

    PARAMETERS
    ----------
    folderpath : str
    -   The path of the folder where the datasets are located.

    analysis_name : str
    -   The name of the analysis folder within the folder given in
        "'folderpath'/settings".

    filetype : str
    -   The file extension, prefixed with a period, e.g. '.json', '.csv'.

    METHODS
    -------
    path
    -   Generates the filepath.
    """

    def __init__(self,
        folderpath: str,
        analysis_name: str,
        filetype: str
        ) -> None:

        self.folderpath = folderpath
        self.analysis_name = analysis_name
        self.filetype = filetype


    def path(self) -> str:
        """Generates a filepath for an object that corresponds to a particular
        analysis spanning multiple recordings sessions.

        RETURNS
        -------
        str
        -   The filepath of the object.
        """

        return os.path.join(self.folderpath, self.analysis_name+self.filetype)
