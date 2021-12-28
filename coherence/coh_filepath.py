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

    def path(self,
        folderpath: str,
        subject: str,
        session: str,
        task: str,
        acquisition: str,
        run: str) -> mne_bids.BIDSPath:

        return mne_bids.BIDSPath(subject=subject,
                                 session=session,
                                 task=task,
                                 acquisition=acquisition,
                                 run=run,
                                 root=folderpath)



class DerivativeFilepath(Filepath):
    """ Generates filepaths for data derivatives (e.g. power, coherence, annotations) based on the BIDS file
    organisation. """


    def subfolders(self) -> str:

        return f"sub-{self.subject}\\ses-{self.session}"


    def filename(self) -> str:

        return f"sub-{self.subject}_ses-{self.session}_task-{self.task}_acq-{self.acquisition}_run-{self.run}_data-{self.datatype}.{self.filetype}"


    def path(self,
        folderpath: str,
        subject: str,
        session: str,
        task: str,
        acquisition: str,
        run: str,
        datatype: str,
        filetype: str) -> str:

        self.folderpath = folderpath
        self.subject = subject
        self.session = session
        self.task = task
        self.acquisition = acquisition
        self.run = run
        self.datatype = datatype
        self.filetype = filetype

        return os.path.join(self.folderpath, self.subfolders(), self.filename())

