"""Functions for dealing with files.

METHODS
-------
generate_raw_fpath
-   Generates an mne_bids.BIDSPath object for loading an mne.io.Raw object.

generate_sessionwise_fpath
-   Generates a filepath for an object that corresponds to an individual
    recording session based on the MNE data-storage filepath structure.

generate_analysiswise_fpath
-   Generates a filepath for an object that corresponds to a particular
    analysis spanning multiple recordings sessions.

check_ftype_present
-   Checks whether a filetype is present in a filepath string based on the
    presence of a period ('.').

identify_ftype
-   Finds what file type of a file is based on the filename extension.
"""

import os
import json
import pickle
from typing import Any, Optional
import mne_bids
from coh_exceptions import (
    MissingFileExtensionError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)


def generate_raw_fpath(
    folderpath: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
) -> mne_bids.BIDSPath:
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

    RETURNS
    -------
    mne_bids.BIDSPath
    -   An mne_bids.BIDSPath object for loading an mne.io.Raw object.
    """

    return mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        root=os.path.join(folderpath, dataset, "rawdata"),
    )


def generate_sessionwise_fpath(
    folderpath: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    group_type: str,
    filetype: str,
) -> str:
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

    RETURNS
    -------
    str
    -   The filepath of the object.
    """

    subfolders = f"{dataset}\\sub-{subject}\\ses-{session}"
    filename = (
        f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_"
        f"{group_type}{filetype}"
    )

    return os.path.join(folderpath, subfolders, filename)


def generate_analysiswise_fpath(
    folderpath: str, analysis_name: str, filetype: str
) -> str:
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

    RETURNS
    -------
    str
    -   The filepath of the object.
    """

    return os.path.join(folderpath, analysis_name + filetype)


def check_ftype_present(fpath: str) -> bool:
    """Checks whether a filetype is present in a filepath string based on the
    presence of a period ('.').

    PARAMETERS
    ----------
    fpath : str
    -   The filepath, including the filename.

    RETURNS
    -------
    ftype_present : bool
    -   Whether or not a filetype is present.
    """

    if "." in fpath:
        ftype_present = True
    else:
        ftype_present = False

    return ftype_present


def identify_ftype(fpath: str) -> str:
    """Finds what file type of a file is based on the filename extension.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath, including the filename and extension.

    RETURNS
    -------
    str
    -   The file type.

    RAISES
    ------
    MissingFileExtensionError
    -   Raised if 'fpath' is missing the filetype extension.
    """

    if not check_ftype_present(fpath):
        raise MissingFileExtensionError(
            "Error when determining the filetype:\nNo filetype can be found in "
            f"the filepath '{fpath}'.\nFilepaths should be in the format "
            "'filename.filetype'."
        )

    return fpath[fpath.rfind(".") + 1 :]


def load_from_json(fpath: str) -> dict:
    """Loads the contents of a json file as a dictionary.

    PARAMETERS
    ----------
    fpath : str
    -   Location of the file to load.

    RETURNS
    -------
    contents : dict
    -   The contents of the file in a dictionary.
    """

    with open(fpath, encoding="utf8") as file:
        contents = json.load(file)

    for key, value in contents.items():
        if value == "INFINITY":
            contents[key] = float("inf")

    return contents


def load_from_pkl(fpath: str) -> Any:
    """Loads the contents of a pkl file.

    PARAMETERS
    ----------
    fpath : str
    -   Location of the file to load.

    RETURNS
    -------
    Any
    -   The contents of the file.
    """

    with open(fpath, "rb") as file:
        return pickle.load(file)


def load_file(
    fpath: str, ftype: Optional[str] = None, verbose: bool = True
) -> Any:
    """Loads the contents of a json or pkl file.

    PARAMETERS
    ----------
    fpath : str
        -   Location where the data should be loaded from.

    ftype : str
    -   The filetype of the data that will be loaded, without the leading
        period. E.g. for loading the file from the json format, this would be
        "json", not ".json".

    verbose : bool; default True
    -   Whether or not to print a note of the loading process.
    """

    if check_ftype_present(fpath) and ftype is not None:
        fpath_ftype = identify_ftype(fpath)
        if fpath_ftype != ftype:
            raise UnidenticalEntryError(
                "Error when trying to save the results of the analysis:\n "
                f"The filetypes in the filepath ({fpath_ftype}) and in the "
                f"requested filetype ({ftype}) do not match."
            )
    elif check_ftype_present(fpath) and ftype is None:
        ftype = identify_ftype(fpath)
    elif not check_ftype_present(fpath) and ftype is not None:
        fpath += ftype
    else:
        raise MissingFileExtensionError(
            "Error when trying to save the results of the analysis:\nNo "
            "filetype is given in the filepath and no filetype has been "
            "specified."
        )

    if ftype == "json":
        contents = load_from_json(fpath=fpath)
    elif ftype == "pkl":
        contents = load_from_pkl(fpath=fpath)
    else:
        raise UnavailableProcessingError(
            f"Error when trying to load the file:\nThe {ftype} format for "
            "loading is not supported."
        )

    if verbose:
        print(f"Loading the contents of the filepath:\n'{fpath}'.")

    return contents
