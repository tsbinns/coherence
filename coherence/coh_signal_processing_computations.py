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
from coh_matlab_functions import mrdivide, reshape


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
    -   Translated into Python from the MATLAB MVGC toolbox v1.0 function
        'cpsd_to_autocov' by Thomas Samuel Binns (@tsbinns).
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

    lags_ifft_shifted_csd = reshape(
        ifft_shifted_csd[:, :, : n_lags + 1],
        (csd_shape[0] ** 2, n_lags + 1),
    )
    signs = [1] * (n_lags + 1)
    signs[1::2] = [x * -1 for x in signs[1::2]]
    sign_matrix = np.tile(np.asarray(signs), (csd_shape[0] ** 2, 1))

    return np.real(
        reshape(
            sign_matrix * lags_ifft_shifted_csd,
            (csd_shape[0], csd_shape[0], n_lags + 1),
        )
    )


def autocovariance_to_full_var(
    autocov: NDArray, enforce_posdef_residuals_cov: bool = False
) -> tuple[NDArray, NDArray]:
    """Computes the full-forward vector autoregressive (VAR) model from an
    autocovariance sequence using Whittle's recursion.

    PARAMETERS
    ----------
    autocov : numpy array
    -   An autocovariance sequence with dimensions [n_signals x n_signals x
        n_lags + 1].

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
    var_coeffs : numpy array
    -   The coefficients of the full forward VAR model with dimensions
        [n_signals x n_signals x n_lags].

    residuals_cov : numpy array
    -   The residuals of the covariance matrix with dimensions [n_signals x
        n_signals].

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
    -   For Whittle's recursion algorithm, see: Whittle P., 1963. Biometrika,
        doi: 10.1093/biomet/50.1-2.129.
    -   Additionally checks that the coefficients are all 'good', i.e. that all
        values are neither 'NaN' nor 'Inf'.
    -   Translated into Python from MATLAB code provided by Stefan Haufe's
        research group by Thomas Samuel Binns (@tsbinns).
    """

    var_coeffs, residuals_cov = whittle_lwr_recursion(
        G=autocov, enforce_coeffs_good=True
    )

    if enforce_posdef_residuals_cov:
        try:
            np.linalg.cholesky(residuals_cov)
        except np.linalg.linalg.LinAlgError as np_error:
            n_lags = np.shape(autocov)[2]
            if n_lags - 1 > 0:
                _, residuals_cov = autocovariance_to_full_var(
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


def whittle_lwr_recursion(
    G: NDArray, enforce_coeffs_good: bool = True
) -> NDArray:
    """Calculates regression coefficients and the residuals' covariance matrix
    from an autocovariance sequence by solving the Yule-Walker equations using
    Whittle's recursive Levinson, Wiggins, Robinson (LWR) algorithm.

    PARAMETERS
    ----------
    G : numpy array
    -   The autocovariance sequence.

    enforce_coeffs_good : bool; default True
    -   Checks that the coefficients of the VAR model are all 'good', i.e. that
        they are all neither 'NaN' or 'Inf', which can happen if the regressions
        are rank-deficient or ill-conditioned.

    RETURNS
    -------
    var_coeffs : numpy array
    -   The coefficients of the full forward VAR model with dimensions
        [n_signals x n_signals x n_lags].

    residuals_cov : numpy array
    -   The residuals of the covariance matrix with dimensions [n_signals x
        n_signals].

    RAISES
    ------
    ValueError
    -   Raised if 'G' does not have three dimensions.
    -   Raised if the first two dimensions of 'G' do not have the same length.
    -   Raised if 'enforce_coeffs_good' is 'True' and the VAR model coefficients
        are not all neither 'NaN' or 'Inf'.

    NOTES
    -----
    -   For Whittle's recursion algorithm, see: Whittle P., 1963. Biometrika,
        doi: 10.1093/biomet/50.1-2.129.
    -   Translated into Python from the MATLAB MVGC toolbox v1.0 function
        'autocov_to_var' by Thomas Samuel Binns (@tsbinns).
    """

    G_shape = np.shape(G)
    if len(G_shape) != 3:
        raise ValueError(
            "The autocovariance sequence must have three dimensions, but has "
            f"{len(G_shape)}."
        )
    if G_shape[0] != G_shape[1]:
        raise ValueError(
            "The autocovariance sequence must have the same first two "
            f"dimensions, but these are {G_shape[0]} and "
            f"{G_shape[1]}, respectively."
        )

    ### Initialise recursion
    n = G_shape[0]  # number of signals
    q = G_shape[2] - 1  # number of lags
    qn = n * q

    G0 = G[:, :, 0]  # covariance
    GF = (
        reshape(G[:, :, 1:], (n, qn)).conj().T
    )  # forward autocovariance sequence
    GB = reshape(
        np.flip(G[:, :, 1:], 2).transpose((0, 2, 1)), (qn, n)
    )  # backward autocovariance sequence

    AF = np.zeros((n, qn))  # forward coefficients
    AB = np.zeros((n, qn))  # backward coefficients

    k = 1  # model order
    r = q - k
    kf = np.arange(k * n)  # forward indices
    kb = np.arange(r * n, qn)  # backward indices

    AF[:, kf] = mrdivide(GB[kb, :], G0)
    AB[:, kb] = mrdivide(GF[kf, :], G0)

    ### Recursion
    for k in np.arange(2, q + 1):
        AAF = mrdivide(
            GB[(r - 1) * n : r * n, :] - np.matmul(AF[:, kf], GB[kb, :]),
            G0 - np.matmul(AB[:, kb], GB[kb, :]),
        )  # DF/VB
        AAB = mrdivide(
            GF[(k - 1) * n : k * n, :] - np.matmul(AB[:, kb], GF[kf, :]),
            G0 - np.matmul(AF[:, kf], GF[kf, :]),
        )  # DB/VF

        AF_previous = AF[:, kf]
        AB_previous = AB[:, kb]

        r = q - k
        kf = np.arange(k * n)
        kb = np.arange(r * n, qn)

        AF[:, kf] = np.hstack((AF_previous - np.matmul(AAF, AB_previous), AAF))
        AB[:, kb] = np.hstack((AAB, AB_previous - np.matmul(AAB, AF_previous)))

    residuals_cov = G0 - np.matmul(AF, GF)
    var_coeffs = reshape(AF, (n, n, q))

    if enforce_coeffs_good:
        if not np.isfinite(var_coeffs).all():
            raise ValueError(
                "The 'good' (i.e. non-NaN and non-infinite) nature of the "
                "VAR model coefficients is being enforced, but the "
                "coefficients are not all finite."
            )

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
    -   Translated into Python from the MATLAB MVGC toolbox v1.0 function
        'bifft' by Thomas Samuel Binns (@tsbinns).
    """

    data_shape = np.shape(data)
    if n_points is None:
        n_points = data_shape[2]

    if len(data_shape) != 3:
        raise ValueError(
            "The cross-spectral density must have three dimensions, but has "
            f"{len(data_shape)} dimensions."
        )

    two_dim_data = reshape(
        data, (data_shape[0] * data_shape[1], data_shape[2])
    ).T
    ifft_data = np.fft.ifft(two_dim_data, n=n_points, axis=0).T

    return reshape(ifft_data, (data_shape[0], data_shape[1], data_shape[2]))


csd = loadmat("coherence\\csd.mat")["CS"]
autocov = csd_to_autocovariance(csd, 20)
var_coeffs, residuals_cov = autocovariance_to_full_var(
    autocov=autocov[:4, :4, :], enforce_posdef_residuals_cov=True
)
print("jeff")
