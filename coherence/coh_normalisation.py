"""Functions for normalising data.

METHODS
-------
find_exclusion_indices
-   Finds the indices of the frequencies to exclude from the normalisation
    calculation.

find_inclusion_indices
-   Finds the indices of the frequencies to include in the normalisation
    calculation.

sort_data_dims
-   Sorts the dimensions of the data being normalised so that the dimension
    being normalised is dimension 0.

restore_data_dims
-   Restores the dimensions of the data to their original format before
    normalisation.

norm_percentage_total
-   Applies percentage total normalisation to the data.
"""

from typing import Union
import numpy as np

from coh_exceptions import EntryLengthError, UnavailableProcessingError
from coh_handle_entries import check_lengths_list_identical


def find_exclusion_indices(
    freqs: list[Union[int, float]],
    line_noise_freq: Union[int, float],
    exclusion_window: Union[int, float],
) -> list[int]:
    """Finds the indices of the frequencies to exclude from the normalisation
    calculation.

    PARAMETERS
    ----------
    freqs : list[int | float]
    -   The frequencies (in Hz) corresponding to the values in 'data'.

    line_noise_freq : int | float
    -   The frequency (in Hz) of the line noise in the data.

    exclusion_window : int | float
    -   The size of the windows (in Hz) to exclude frequencies around the line
        noise and harmonic frequencies from the calculations of what to
        normalise the data by.
    -   If 0, no frequencies are excluded.
    -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is 10,
        the results from 45 - 55 Hz would be ommited.

    RETURNS
    -------
    exclusion_indices : list[int]
    -   The indices of the entries in 'freqs' to exclude from the normalisation
        calculation.

    RAISES
    ------
    ValueError
    -   Raised if the 'exclusion_window' is less than 0.
    """

    if exclusion_window < 0:
        raise ValueError(
            "Error when finding indices of data to exclude:\nThe exclusion "
            "window must be greater than 0."
        )

    if exclusion_window != 0:
        half_window = exclusion_window / 2
        exclusion_indices = []
        bad_freqs = np.arange(
            start=0, stop=freqs[-1] + line_noise_freq, step=line_noise_freq
        )
        for bad_freq in bad_freqs:
            for freq_i, freq in enumerate(freqs):
                if (
                    freq >= bad_freq - half_window
                    and freq <= bad_freq + half_window
                ):
                    exclusion_indices.append(freq_i)
    else:
        exclusion_indices = []

    return exclusion_indices


def find_inclusion_indices(
    freqs: list[Union[int, float]],
    line_noise_freq: Union[int, float],
    exclusion_window: Union[int, float],
) -> list[int]:
    """Finds the indices of the frequencies to include in the normalisation
    calculation.

    PARAMETERS
    ----------
    freqs : list[int | float]
    -   The frequencies (in Hz) corresponding to the values in 'data'.

    line_noise_freq : int | float
    -   The frequency (in Hz) of the line noise in the data.

    exclusion_window : int | float
    -   The size of the windows (in Hz) to exclude frequencies around the line
        noise and harmonic frequencies from the calculations of what to
        normalise the data by.
    -   If 0, no frequencies are excluded.
    -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is 10,
        the results from 45 - 55 Hz would be ommited.

    RETURNS
    -------
    inclusion_indices : list[int]
    -   The indices of the entries in 'freqs' to include in the normalisation
        calculation.
    """

    exclusion_indices = find_exclusion_indices(
        freqs=freqs,
        line_noise_freq=line_noise_freq,
        exclusion_window=exclusion_window,
    )

    return [i for i in range(len(freqs)) if i not in exclusion_indices]


def sort_data_dims(
    data: np.ndarray,
    data_dims: list[str],
    within_dim: str,
) -> tuple[np.ndarray, list[str]]:
    """Sorts the dimensions of the data being normalised so that the dimension
    being normalised is dimension 0.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   The data to normalise.

    data_dims : list[str]
    -   Descriptions of the data dimensions.

    within_dim : str
    -   The dimension to apply the normalisation within.
    -   E.g. if the data has dimensions "channels" and "frequencies", setting
        'within_dims' to "channels" would normalise the data across the
        frequencies within each channel.

    RETURNS
    -------
    data : numpy ndarray
    -   The data with the dimension being normalised as the 0th dimension.

    sorted_data_dims : list[str]
    -   The dimensions of the sorted data.
    """

    within_dim_i = data_dims.index(within_dim)
    if within_dim_i != 0:
        sorted_data_dims = [within_dim].extend(
            [dim for dim in data_dims if dim != within_dim]
        )
        transposition_indices = [
            data_dims.index(dim) for dim in sorted_data_dims
        ]
        data = np.transpose(data, transposition_indices)
    else:
        sorted_data_dims = data_dims

    return data, sorted_data_dims


def restore_data_dims(
    data: np.ndarray, current_dims: list[str], restore_dims: list[str]
) -> np.ndarray:
    """Restores the dimensions of the data to their original format before
    normalisation.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   The data whose dimensions will be restored.

    current_dims : list[str]
    -   The dimensions of 'data'.

    restore_dims : list[str]
    -   The dimensions of 'data' to restore.

    RETURNS
    -------
    data : numpy ndarray
    -   The data with restored dimensions.
    """

    identical, lengths = check_lengths_list_identical(
        to_check=[data.shape, current_dims, restore_dims]
    )
    if not identical:
        raise EntryLengthError(
            "Error when restoring the dimensions of the data after "
            "normalisation:\nThe lengths of the actual data dimensions "
            f"({lengths[0]}), specified data dimensions ({lengths[1]}), and "
            f"desired data dimensions ({lengths[2]}) must match."
        )

    return np.transpose(data, [current_dims.index(dim) for dim in restore_dims])


def norm_percentage_total(
    data: np.ndarray,
    freqs: list[Union[int, float]],
    data_dims: list[str],
    within_dim: str,
    line_noise_freq: Union[int, float],
    exclusion_window: Union[int, float],
) -> np.ndarray:
    """Applies percentage total normalisation to the data.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   The data to normalise.

    freqs : list[int | float]
    -   The frequencies (in Hz) corresponding to the values in 'data'.

    data_dims : list[str]
    -   Descriptions of the data dimensions.

    within_dim : str
    -   The dimension to apply the normalisation within.
    -   E.g. if the data has dimensions "channels" and "frequencies", setting
        'within_dims' to "channels" would normalise the data across the
        frequencies within each channel.
    -   Currently, normalising only two-dimensional data is supported.

    line_noise_freq : int | float
    -   The frequency (in Hz) of the line noise in the data.

    exclusion_window : int | float
    -   The size of the windows (in Hz) to exclude frequencies around the line
        noise and harmonic frequencies from the calculations of what to
        normalise the data by.
    -   If 0, no frequencies are excluded.
    -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is 10,
        the results from 45 - 55 Hz would be ommited.
    """

    if len(data_dims) > 2 or len(data.shape) > 2:
        raise UnavailableProcessingError(
            "Error when percentage-total normalising the data:\nOnly "
            "two-dimensional data can be normalised."
        )

    data, new_data_dims = sort_data_dims(
        data=data, data_dims=data_dims, within_dim=within_dim
    )

    inclusion_idcs = find_inclusion_indices(
        freqs=freqs,
        line_noise_freq=line_noise_freq,
        exclusion_window=exclusion_window,
    )

    for data_i in range(data.shape[0]):
        data[data_i] = (
            data[data_i] / np.sum(data[data_i][inclusion_idcs])
        ) * 100

    return restore_data_dims(
        data=data, current_dims=new_data_dims, restore_dims=data_dims
    )
