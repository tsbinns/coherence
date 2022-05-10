"""Methods for creating and handling objects."""

from typing import Union
import mne
from numpy.typing import NDArray
from coh_handle_entries import rearrange_axes


def create_extra_info(data: dict) -> dict[dict]:
    """Create a dictionary for holding additional information used in Signal
    objects.

    PARAMETERS
    ----------
    data : dict
    -   Data dictionary in the same format as that derived from a Signal object
        to extract the extra information from.

    RETURNS
    -------
    extra_info : dict[dict]
    -   Additional information extracted from the data dictionary.
    """

    ch_dict_keys = [
        "ch_regions",
        "ch_hemispheres",
        "ch_reref_types",
        "ch_epoch_orders",
    ]
    extra_info_keys = [
        "ch_regions",
        "ch_hemispheres",
        "ch_reref_types",
        "ch_epoch_orders",
        "metadata",
    ]
    extra_info = {}
    for key in extra_info_keys:
        if key in data.keys():
            if key in ch_dict_keys:
                extra_info[key] = {
                    name: data[key][i]
                    for i, name in enumerate(data["ch_names"])
                }
            else:
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
) -> tuple[Union[mne.io.Raw, mne.Epochs], list[str]]:
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

    new_dimensions : list[str]
    -   Names of the new axes in 'data'.
    -   ["epochs", "channels", "timepoints"] if 'data_object' is an MNE Epochs
        object.
    -   ["channels", "timepoints"] if 'data_object' is an MNE Raw object.
    """

    data_info = create_mne_data_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq,
        subject_info=subject_info,
        verbose=verbose,
    )

    if "epochs" in data_dimensions:
        new_dimensions = ["epochs", "channels", "timepoints"]
        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=new_dimensions,
        )
        data_object = mne.EpochsArray(data=data, info=data_info)
    else:
        new_dimensions = ["channels", "timepoints"]
        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=new_dimensions,
        )
        data_object = mne.io.RawArray(data=data, info=data_info)

    if ch_coords is not None:
        data_object._set_channel_positions(pos=ch_coords, names=ch_names)

    return data_object, new_dimensions


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
