"""Methods for computing connectivity metrics.

METHODS
-------
multivariate_connectivity
-   Method for directing to different multivariate connectivity methods.

multivariate_interaction_measure
-   Computes the multivariate interaction measure between two groups of signals.

max_imaginary_coherence
-   Computes the maximised imaginary coherence between two groups of signals.

multivariate_connectivity_compute_E
-   Computes 'E' as the imaginary part of the transformed connectivity matrix
    'D' derived from the original connectivity matrix 'C' between the signals in
    groups A and B.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.io import loadmat


def multivariate_connectivity(
    data: NDArray, method: str, n_group_a: int, n_group_b: int
) -> NDArray:
    """Method for directing to different multivariate connectivity methods.

    PARAMETERS
    ----------
    data : numpy array
    -   A three-dimensional matrix with dimensions [nodes x nodes x frequencies]
        containing coherency values for all possible connections between signals
        in two groups: A and B. The number of nodes is equal to the number of
        signals in group A ('n_group_a') plus the number of signals in group B
        ('n_group_b').

    method : str
    -   The multivariate connectivity metric to compute.
    -   Supported inputs are: "mim" for multivariate interaction measure; "mic"
        for maximised imaginary coherence.

    n_group_a : int
    -   Number of signals in group A. Entries in the first two dimensions of
        'data' from '0 : n_group_a' are taken as the coherency values for
        signals in group A.

    n_group_b : int
    -   Number of signals in group B. Entries in the first two dimensions of
        'data' from 'n_group_a : n_group_b' are taken as the coherency values
        for signals in group B.

    RETURNS
    -------
    results : NDArray
    -   Vector containing the computed multivariate connectivity values with
        length equal to the number of frequencies in 'data'.

    RAISES
    ------
    NotImplementedError
    -   Raised if the requested method is not supported.
    """

    supported_methods = ["mim", "mic"]
    if method not in supported_methods:
        raise NotImplementedError(
            "Error when computing multivariate connectivity metrics:\nThe "
            f"method '{method}' is not supported. Supported methods are "
            f"{supported_methods}."
        )

    if method == "mim":
        results = multivariate_interaction_measure(
            data=data, n_group_a=n_group_a, n_group_b=n_group_b
        )
    else:
        results = max_imaginary_coherence(
            data=data, n_group_a=n_group_a, n_group_b=n_group_b
        )

    return results


def multivariate_interaction_measure(
    data: NDArray, n_group_a: int, n_group_b: int
) -> NDArray:
    """Computes the multivariate interaction measure between two groups of
    signals.

    data : numpy array
    -   Coherency values between all possible connections of two groups of
        signals, A and B, across frequencies. Has the dimensions [signals x
        signals x frequencies].

    n_group_a : int
    -   Number of signals in group A. Entries in the first two dimensions of
        'data' from '0 : n_group_a' are taken as the coherency values for
        signals in group A.

    n_group_b : int
    -   Number of signals in group B. Entries in the first two dimensions of
        'data' from 'n_group_a : n_group_b' are taken as the coherency values
        for signals in group B.

    RETURNS
    -------
    mim : numpy array
    -   One-dimensional array containing a connectivity value between signal
        groups A and B for each frequency.

    RAISES
    ------
    ValueError
    -   Raised if the data is not a three-dimensional array.
    -   Raised if the first two dimensions of 'data' is not a square matrix with
        lengths equal to the combined number of signals in groups A and B.

    NOTES
    -----
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    -   All equations were written in MATLAB by Franziska Pellegrini working in
        the group of Stefan Haufe, which were then translated into Python by
        Thomas Samuel Binns working in the group of Wolf-Julian Neumann.
    """

    if len(np.shape(data)) != 3:
        raise ValueError(
            "Error when computing MIC:\nThe data must be a three-dimensional "
            "array containing connectivity values across frequencies, but the "
            f"data has {len(np.shape(data))} dimensions."
        )
    n_signals = n_group_a + n_group_b
    if (n_signals, n_signals) != np.shape(data)[0:2]:
        raise ValueError(
            "Error when calculating the multivariate interaction measure:\nThe "
            f"data for each frequency must be a [{n_signals} x {n_signals}] "
            "square matrix containing all connectivities between the "
            f"{n_group_a} signals in group A and the {n_group_b} signals in "
            f"group B, but it is a [{np.shape(data)[0]} x {np.shape(data)[1]}] "
            "matrix."
        )

    n_freqs = np.shape(data)[2]
    mim = np.empty(n_freqs)
    for freq_i in range(n_freqs):
        # Equations 2-4
        E = multivariate_connectivity_compute_E(
            data=data[:, :, freq_i], n_group_a=n_group_a
        )

        # Equation 14
        mim[freq_i] = np.trace(np.matmul(E, E.T))

    return mim


def max_imaginary_coherence(
    data: NDArray, n_group_a: int, n_group_b: int
) -> NDArray:
    """Computes the maximised imaginary coherence between two groups of signals.

    data : numpy array
    -   Coherency values between all possible connections of two groups of
        signals, A and B, across frequencies. Has the dimensions [signals x
        signals x frequencies].

    n_group_a : int
    -   Number of signals in group A. Entries in the first two dimensions of
        'data' from '0 : n_group_a' are taken as the coherency values for
        signals in group A.

    n_group_b : int
    -   Number of signals in group B. Entries in the first two dimensions of
        'data' from 'n_group_a : n_group_b' are taken as the coherency values
        for signals in group B.

    RETURNS
    -------
    mic : numpy array
    -   One-dimensional array containing a connectivity value between signal
        groups A and B for each frequency.

    RAISES
    ------
    ValueError
    -   Raised if the data is not a three-dimensional array.
    -   Raised if the first two dimensions of 'data' is not a square matrix with
        lengths equal to the combined number of signals in groups A and B.

    NOTES
    -----
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    -   All equations were written in MATLAB by Franziska Pellegrini working in
        the group of Stefan Haufe, which were then translated into Python by
        Thomas Samuel Binns working in the group of Wolf-Julian Neumann.
    """

    if len(np.shape(data)) != 3:
        raise ValueError(
            "Error when computing MIC:\nThe data must be a three-dimensional "
            "array containing connectivity values across frequencies, but the "
            f"data has {len(np.shape(data))} dimensions."
        )
    n_signals = n_group_a + n_group_b
    if (n_signals, n_signals) != np.shape(data)[0:2]:
        raise ValueError(
            "Error when calculating the multivariate interaction measure:\nThe "
            f"data for each frequency must be a [{n_signals} x {n_signals}] "
            "square matrix containing all connectivities between the "
            f"{n_group_a} signals in group A and the {n_group_b} signals in "
            f"group B, but it is a [{np.shape(data)[0]} x {np.shape(data)[1]}] "
            "matrix."
        )

    n_freqs = np.shape(data)[2]
    mic = np.empty(n_freqs)
    for freq_i in range(n_freqs):
        # Equations 2-4
        E = multivariate_connectivity_compute_E(
            data=data[:, :, freq_i], n_group_a=n_group_a
        )

        # Weights for signals in the groups
        w_a, V_a = np.linalg.eig(np.matmul(E, E.T))
        w_b, V_b = np.linalg.eig(np.matmul(E.T, E))
        alpha = V_a[:, w_a.argmax()]
        beta = V_b[:, w_b.argmax()]

        # Equation 7
        mic[freq_i] = np.abs(
            np.matmul(alpha.T, np.matmul(E, beta))
            / np.linalg.norm(alpha)
            * np.linalg.norm(beta)
        )

    return mic


def multivariate_connectivity_compute_E(
    data: NDArray, n_group_a: int
) -> NDArray:
    """Computes 'E' as the imaginary part of the transformed connectivity matrix
    'D' derived from the original connectivity matrix 'C' between the signals in
    groups A and B.
    -   Designed for use with the methods 'max_imaginary_coherence' and
        'multivariate_interaction_measure'.

    data : numpy array
    -   Coherency values between all possible connections of two groups of
        signals, A and B, for a single frequency. Has the dimensions [signals x
        signals].

    n_group_a : int
    -   Number of signals in group A. Entries in both dimensions of 'data' from
        '0 : n_group_a' are taken as the coherency values for signals in group
        A. Entries from 'n_group_a : end' are taken as the coherency values for
        signals in group B.

    RETURNS
    -------
    E : numpy array
    -   The imaginary part of the transformed connectivity matrix 'D' between
        signals in groups A and B.

    NOTES
    -----
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    -   All equations were written in MATLAB by Franziska Pellegrini working in
        the group of Stefan Haufe, which were then translated into Python by
        Thomas Samuel Binns working in the group of Wolf-Julian Neumann.
    """

    # Equation 2
    C_aa = data[0:n_group_a, 0:n_group_a]
    C_ab = data[0:n_group_a, n_group_a:]
    C_bb = data[n_group_a:, n_group_a:]
    C_ba = data[n_group_a:, 0:n_group_a]
    C = np.vstack((np.hstack((C_aa, C_ab)), np.hstack((C_ba, C_bb))))

    # Equation 3
    T = np.zeros(np.shape(C))
    T[0:n_group_a, 0:n_group_a] = fractional_matrix_power(np.real(C_aa), -0.5)
    T[n_group_a:, n_group_a:] = fractional_matrix_power(np.real(C_bb), -0.5)

    # Equation 4
    D = np.matmul(T, np.matmul(C, T))

    # 'E' as the imaginary part of 'D' between groups A and B
    E = np.imag(D[0:n_group_a, n_group_a:])

    return E


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
    -   Raised if the first two dimensions of 'csd' are not identical.
    -   Raised if 'n_lags' is greater than (n_freqs - 1) * 2.

    NOTES
    -----
    -   Translated into Python from the MATLAB MVGC toolbox function
        "cpsd_to_autocov" by Thomas Samuel Binns of Wolf-Julian Neumann's group.
    """

    ### Input sorting and checking
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

    ### Computations
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


def autocov_to_autoreg_model(
    autocov: NDArray, enforce_posdef_residuals_cov: bool = False
) -> None:
    """Computes an autoregressive model from an autocovariance seuqence.

    PARAMETERS
    ----------
    autocov : numpy array

    enforce_posdef_residuals_cov : bool; default False
    -   Whether or not to make sure that the residuals' covariance matrix is
        positive-definite.
    -   This is checked using the Cholesky decomposition, and therefore assumes
        that the matrix is symmetric.
    -   If this is 'True' and the residuals' covariance matrix is found to not
        be positive-definite, the conversion to an autoregressive model will be
        run again using autcov[:,:,-1] until a positive-definite matrix is
        found. If this is continually re-run until the third dimension of
        'autocov' has length 1 and no positive-definite matrix has still been
        found, an error is raised.

    RETURNS
    -------

    RAISES
    ------
    ValueError
    -   Raised if no positive-definite residuals' covariance matrix can be found
        and 'enforce_posdef_residuals_cov' is 'True'.

    NOTES
    -----
    """

    n_lags = np.shape(autocov)[2]
    if enforce_posdef_residuals_cov:
        try:
            np.linalg.cholesky(residuals_cov)
        except np.linalg.linalg.LinAlgError as np_error:
            if n_lags - 1 > 0:
                _, residuals_cov = autocov_to_autoreg_model(
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


csd = loadmat("coherence\\csd.mat")["S"]
autocov = csd_to_autocovariance(csd, 20)
