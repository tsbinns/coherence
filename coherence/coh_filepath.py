import os
from abc import ABC, abstractmethod
import mne_bids




class Filepath(ABC):
    """ Generates filepaths based on the BIDS file organisation. """

    @abstractmethod
    def path():
        pass



class RawFilepath(Filepath):
    """ Generates filepaths for raw data as mne_bids.BIDSPath objects. """

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

        return mne_bids.BIDSPath(
            subject=self.subject, session=self.session, task=self.task,
            acquisition=self.acquisition, run=self.run,
            root=os.path.join(self.folderpath, self.dataset, 'rawdata')
        )



class DataWiseFilepath(Filepath):

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


    def _subfolders(self) -> str:

        return (f"{self.dataset}\\sub-{self.subject}\\ses-{self.session}\\"
            f"{self.group_type}")


    def _filename(self) -> str:

        return (f"sub-{self.subject}_ses-{self.session}_task-{self.task}_"
            f"acq-{self.acquisition}_run-{self.run}_{self.group_type}"
            f"{self.filetype}"
        )


    def path(self) -> str:

        return os.path.join(
            self.folderpath, self._subfolders(), self._filename()
        )


class AnalysisWiseFilepath(Filepath):

    def __init__(self,
        folderpath: str,
        analysis_name: str,
        filetype: str
        ) -> None:

        self.folderpath = folderpath
        self.analysis_name = analysis_name
        self.filetype = filetype

    def path(self) -> str:

        return os.path.join(self.folderpath, self.analysis_name+self.filetype)