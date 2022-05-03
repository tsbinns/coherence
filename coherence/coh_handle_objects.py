"""Methods for creating and handling objects."""

from typing import Union
import mne
from numpy.typing import NDArray
from coh_handle_entries import rearrange_axes


def create_extra_info(data: dict) -> dict:
    """Create a dictionary for holding additional information used in Signal
    objects.

    PARAMETERS
    ----------
    data : dict
    -   Data dictionary in the same format as that derived from a Signal object
        to extract the extra information from.

    RETURNS
    -------
    extra_info : dict
    -   Additional information extracted from the data dictionary.
    """

    extra_info = {
        "ch_regions": None,
        "ch_hemispheres": None,
        "reref_types": None,
        "metadata": None,
    }
    for key in extra_info.keys():
        extra_info[key] = data[key]

    return extra_info


def create_mne_data_object(
    data: Union[list, NDArray],
    data_dimensions: list[str],
    ch_names: list[str],
    ch_types: list[str],
    sfreq: Union[int, float],
    ch_coords: Union[list[list[Union[int, float]]], None] = None,
    subject_info: Union[dict, None] = None,
    verbose: bool = True,
) -> Union[mne.io.Raw, mne.Epochs]:
    """Creates an MNE Raw or Epochs object, depending on the structure of the
    data.

    PARAMETERS
    ----------
    data : list | numpy array
    -   The data of the object.

    data_dimensions : list[str]
    -   Names of axes in 'data'.
    -   If "epochs" is in the dimensions, an MNE Epochs object is created,
        otherwise an MNE Raw object is created.
    -   MNE Epochs objects expect data to be in the dimensions ["epochs",
        "channels", "timepoints"], which the data axes are rearranged to if they
        do not match.
    -   MNE Raw objects expect data to be in the dimensions ["channels",
        "timepoints"], which the data axes are rearranged to if they do not
        match.

    ch_names : list[str]
    -   Names of the channels.

    ch_types : list[str]
    -   Types of the channels.

    sfreq : Union[int, float]
    -   Sampling frequency of the data.

    ch_coords : list[list[int | float]] | None; default None
    -   Coordinates of the channels.

    subject_info : dict | None; default None
    -   Information about the subject from which the data was collected.

    verbose : bool; default True
    -   Verbosity setting of the generated MNE object.

    RETURNS
    -------
    data_object : MNE Raw | MNE Epochs
    -   The generated MNE object containing the data.
    """

    data_info = create_mne_data_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq,
        subject_info=subject_info,
        verbose=verbose,
    )

    if "epochs" in data_dimensions:
        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=["epochs", "channels", "timepoints"],
        )
        data_object = mne.EpochsArray(data=data, info=data_info)
    else:
        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=["channels", "timepoints"],
        )
        data_object = mne.io.RawArray(data=data, info=data_info)

    if ch_coords is not None:
        data_object._set_channel_positions(pos=ch_coords, names=ch_names)

    return data_object


def create_mne_data_info(
    ch_names: list[str],
    ch_types: list[str],
    sfreq: Union[int, float],
    subject_info: Union[dict, None] = None,
    verbose: bool = True,
) -> mne.Info:
    """

    PARAMETERS
    ----------
    ch_names : list[str]
    -   Names of the channels.

    ch_types : list[str]
    -   Types of the channels.

    sfreq : Union[int, float]
    -   Sampling frequency of the data.

    subject_info : dict | None; default None
    -   Information about the subject from which the data was collected.

    verbose : bool; default True
    -   Verbosity setting of the generated MNE object.

    RETURNS
    -------
    data_info : MNE Info
    -   Information about the data.
    """

    data_info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=verbose
    )
    if subject_info is not None:
        data_info["subject_info"] = subject_info

    return data_info
