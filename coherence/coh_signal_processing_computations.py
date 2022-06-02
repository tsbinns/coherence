"""Methods for performing signal processing computations.

METHODS
-------
block_ifft
-   Performs a 'block' inverse fast Fourier transform on the data, involving an
    n-point inverse Fourier transform.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat


def csd_to_autocovariance(
    csd: NDArray, n_lags: Union[int, None] = None
) -> NDArray:
    """Computes the autocovariance sequence from the cross-spectral density.

    PARAMETERS
    ----------
    csd : numpy array
    -   Three-dimensional matrix of the cross-spectral density.
    -   Expects a matrix with dimensions [n_signals x n_signals x n_freqs],
        where the third dimension corresponds to different frequencies.

    n_lags : int | None; default None
    -   Number of autocovariance lags to calculate.
    -   If 'None', the number of lags is the number of frequencies in the
        cross-spectra minus two.

    RETURNS
    -------
    numpy array
    -   The computed autocovariance sequence with dimensions [n_signals x
        n_signals x n_lags plus one]

    RAISES
    ------
    ValueError
    -   Raised if 'csd' is not three-dimensional.
    -   Raised if the lengths of the first two dimensions of 'csd' are not
        identical.
    -   Raised if 'n_lags' is greater than (n_freqs - 1) * 2.

    NOTES
    -----
    -   Translated into Python from the MATLAB MVGC toolbox function
        "cpsd_to_autocov" by Thomas Samuel Binns of Wolf-Julian Neumann's group.
    """

    csd_shape = np.shape(csd)
    if len(csd_shape) != 3:
        raise ValueError(
            "The cross-spectral density must have three dimensions, but "
            f"has {len(csd_shape)}."
        )
    if csd_shape[0] != csd_shape[1]:
        raise ValueError(
            "The cross-spectral density must have the same first two "
            f"dimensions, but these are {csd_shape[0]} and {csd_shape[1]}, "
            "respectively."
        )
    n_freqs = csd_shape[2]
    freq_res = n_freqs - 1
    if n_lags is None:
        n_lags = freq_res - 1
    if n_lags > freq_res * 2:
        raise ValueError(
            f"The number of lags ({n_lags}) cannot be greater than the "
            "frequency resolution of the cross-spectral density "
            f"({freq_res})."
        )

    circular_shifted_csd = np.concatenate(
        [np.flip(np.conj(csd[:, :, 1:]), axis=2), csd[:, :, :-1]], axis=2
    )
    ifft_shifted_csd = block_ifft(
        data=circular_shifted_csd, n_points=freq_res * 2
    )

    lags_ifft_shifted_csd = np.reshape(
        ifft_shifted_csd[:, :, : n_lags + 1],
        (csd_shape[0] ** 2, n_lags + 1),
    )
    signs = [1] * (n_lags + 1)
    signs[1::2] = [x * -1 for x in signs[1::2]]
    sign_matrix = np.tile(np.asarray(signs), (csd_shape[0] ** 2, 1))

    return np.real(
        np.reshape(
            sign_matrix * lags_ifft_shifted_csd,
            (csd_shape[0], csd_shape[0], n_lags + 1),
        )
    )


def autocovariance_to_full_var(
    autocov: NDArray, enforce_posdef_residuals_cov: bool = False
) -> None:
    """Computes the full-forward vector autoregressive model from an
    autocovariance sequence.

    PARAMETERS
    ----------
    autocov : numpy array
    -   An autocovariance sequence with dimensions [n_signals x n_signals x
        n_lags].

    enforce_posdef_residuals_cov : bool; default False
    -   Whether or not to make sure that the residuals' covariance matrix is
        positive-definite.
    -   If this is 'True' and the residuals' covariance matrix is found to not
        be positive-definite, the conversion to an autoregressive model will be
        run again using autcov[:,:,-1] until a positive-definite matrix is
        found. If this is continually re-run until the third dimension of
        'autocov' has length 1 and no positive-definite matrix has still been
        found, an error is raised.
    -   This is checked using the Cholesky decomposition, and therefore assumes
        that the matrix is symmetric.

    RETURNS
    -------

    RAISES
    ------
    ValueError
    -   Raised if 'autocov' is not three-dimensional.
    -   Raised if the lengths of the first two dimensions of 'autocov' are not
        identical.
    -   Raised if no positive-definite residuals' covariance matrix can be found
        and 'enforce_posdef_residuals_cov' is 'True'.

    NOTES
    -----
    """

    csd_shape = np.shape(csd)
    if len(csd_shape) != 3:
        raise ValueError(
            "The cross-spectral density must have three dimensions, but "
            f"has {len(csd_shape)}."
        )
    if csd_shape[0] != csd_shape[1]:
        raise ValueError(
            "The cross-spectral density must have the same first two "
            f"dimensions, but these are {csd_shape[0]} and {csd_shape[1]}, "
            "respectively."
        )

    n_lags = np.shape(autocov)[2]
    if enforce_posdef_residuals_cov:
        try:
            np.linalg.cholesky(residuals_cov)
        except np.linalg.linalg.LinAlgError as np_error:
            if n_lags - 1 > 0:
                _, residuals_cov = autocov_to_full_var(
                    autocov=autocov[:, :, : n_lags - 1],
                    enforce_posdef_residuals_cov=True,
                )
            else:
                raise ValueError(
                    "The positive-definite nature of the residuals' covariance "
                    "matrix is being enforced, however no positive-definite "
                    "matrix can be found."
                ) from np_error

    return var_coeffs, residuals_cov


def block_ifft(data: NDArray, n_points: Union[int, None] = None) -> NDArray:
    """Performs a 'block' inverse fast Fourier transform on the data, involving
    an n-point inverse Fourier transform.

    PARAMETERS
    ----------
    data : numpy array
    -   A three-dimensional matrix on which the inverse Fourier transform will
        be conducted, where the third dimension is assumed to correspond to
        different frequencies.

    n_points : int | None; default None
    -   The number of points to use for the inverse Fourier transform.
    -   If 'None', the numbe of frequencies in the data (i.e. the length of the
        third dimension) is used.

    RETURNS
    -------
    numpy array
    -   A three-dimensional matrix of the transformed data.

    RAISES
    ------
    ValueError
    -   Raised if 'data' does not have three dimensions.

    NOTES
    -----
    -   Translated into Python from the MATLAB MVGC toolbox function "bifft" by
        by Thomas Samuel Binns of Wolf-Julian Neumann's group.
    """

    data_shape = np.shape(data)
    if n_points is None:
        n_points = data_shape[2]

    if len(data_shape) != 3:
        raise ValueError(
            "The cross-spectral density must have three dimensions, but has "
            f"{len(data_shape)} dimensions."
        )

    two_dim_data = np.reshape(
        data, (data_shape[0] * data_shape[1], data_shape[2])
    ).T
    ifft_data = np.fft.ifft(two_dim_data, n=n_points, axis=0).T

    return np.reshape(ifft_data, (data_shape[0], data_shape[1], data_shape[2]))


csd = loadmat("coherence\\csd.mat")["CS"]
autocov = csd_to_autocovariance(csd, 20)
print("jeff")
