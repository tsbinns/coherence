"""Methods for performing signal processing computations.

METHODS
-------
block_ifft
-   Performs a 'block' inverse fast Fourier transform on the data, involving an
    n-point inverse Fourier transform.
"""

from typing import Union
import numpy as np
from scipy import linalg as spla
from numpy.typing import NDArray
from coh_handle_entries import check_posdef
from coh_matlab_functions import mrdivide, reshape, kron


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
        n_signals x (n_lags + 1)]

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
    csd_shape = csd.shape
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
    G: NDArray, enforce_posdef_residuals_cov: bool = False
) -> tuple[NDArray, NDArray]:
    """Computes the full vector autoregressive (VAR) model from an
    autocovariance sequence using Whittle's recursion.

    PARAMETERS
    ----------
    G : numpy array
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
    AF : numpy array
    -   The coefficients of the full forward VAR model with dimensions
        [n_signals x n_signals x n_lags].

    V : numpy array
    -   The residuals' covariance matrix with dimensions [n_signals x
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
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Stefan Haufe's research group.
    """
    AF, V = whittle_lwr_recursion(G=G, enforce_coeffs_good=True)

    if enforce_posdef_residuals_cov:
        try:
            np.linalg.cholesky(V)
        except np.linalg.linalg.LinAlgError as np_error:
            n_lags = G.shape[2]
            if n_lags - 1 > 1:
                _, V = autocovariance_to_full_var(
                    G=G[:, :, : n_lags - 1],
                    enforce_posdef_residuals_cov=True,
                )
            else:
                raise ValueError(
                    "The positive-definite nature of the residuals' covariance "
                    "matrix is being enforced, however no positive-definite "
                    "matrix can be found."
                ) from np_error

    return AF, V


def whittle_lwr_recursion(
    G: NDArray, enforce_coeffs_good: bool = True
) -> NDArray:
    """Calculates regression coefficients and the residuals' covariance matrix
    from an autocovariance sequence by solving the Yule-Walker equations using
    Whittle's recursive Levinson, Wiggins, Robinson (LWR) algorithm.

    PARAMETERS
    ----------
    G : numpy array
    -   The autocovariance sequence with dimensions [n_signals x n_signals x
        (n_lags + 1)].

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
    -   The residuals' covariance matrix with dimensions [n_signals x
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
    -   For Whittle's recursion algorithm, see: Whittle P., 1963, Biometrika,
        DOI: 10.1093/biomet/50.1-2.129.
    -   Translated into Python from the MATLAB MVGC toolbox v1.0 function
        'autocov_to_var' by Thomas Samuel Binns (@tsbinns).
    """
    G_shape = G.shape
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
    # AF[:, kf] = np.matmul(GB[kb, :], np.linalg.inv(G0))
    # AB[:, kb] = np.matmul(GB[kf, :], np.linalg.inv(G0))

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

    V = G0 - np.matmul(AF, GF)
    AF = reshape(AF, (n, n, q))

    if enforce_coeffs_good:
        if not np.isfinite(AF).all():
            raise ValueError(
                "The 'good' (i.e. non-NaN and non-infinite) nature of the "
                "VAR model coefficients is being enforced, but the "
                "coefficients are not all finite."
            )

    return AF, V


def var_to_ss_params(AF: NDArray, V: NDArray):
    """Computes innovations-form parameters for a state-space vector
    autoregressive (VAR) model from a VAR model's coefficients and residuals'
    covariance matrix using Aoki's method.

    PARAMETERS
    ----------
    AF : numpy array
    -   The coefficients of the full VAR model with dimensions [n_signals x
        n_signals*n_lags].

    V : numpy array
    -   The residuals' covariance matrix with dimensions [n_signals x
        n_signals].

    RETURNS
    -------
    A : numpy array
    -   ???

    K : numpy array
    -   ???

    NOTES
    -----
    -   Aoki's method for computing innovations-form parameters for a
        state-space VAR model allows for zero-lag coefficients.
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Stefan Haufe's research group.
    """
    AF_shape = AF.shape
    if len(AF_shape) != 2:
        raise ValueError(
            "The VAR model coefficients must have two dimensions, but has "
            f"{len(AF_shape)}."
        )
    V_shape = V.shape
    if len(V_shape) != 2:
        raise ValueError(
            "The VAR model's residual's covariance matrix must have two "
            f"dimensions, but has {len(V_shape)}."
        )
    if V_shape[0] != V_shape[1]:
        raise ValueError(
            "The VAR model's residual's covariance matrix must have the same "
            f"first two dimensions, but these are {V_shape[0]} and "
            f"{V_shape[1]}, respectively."
        )

    m = AF.shape[0]  # number of signals
    p = AF.shape[1] // m  # number of autoregressive lags

    Ip = np.eye(m * p)
    A = np.vstack((AF, Ip[: (len(Ip) - m), :]))
    K = np.vstack((np.eye(m), np.zeros(((m * (p - 1)), m))))

    # O = discrete_lyapunov(A=A, Q=-np.matmul(K, np.matmul(V, K.conj().T)))
    # lambda_0 = (
    #    np.matmul(AF, np.matmul(O, AF.conj().T)) + V
    # )  # variance of the process

    return A, K


def ss_params_to_gc(
    A: NDArray,
    C: NDArray,
    K: NDArray,
    V: NDArray,
    freqs: list[Union[int, float]],
    seeds: list[int],
    targets: list[int],
) -> NDArray:
    """Computes frequency-domain Granger causality from innovations-form
    parameters for a state-space vector autoregressive (VAR) model.

    PARAMETERS
    ----------
    A : numpy array
    -   ??? Matrix of innovations-form state space VAR model parameters with
        dimensions [m x m].

    C : numpy array
    -   Coeffients innovations-form state space VAR model parameters with
        dimensions [n x m], where 'n' is the number of signals and 'm' the
        number of signals times the number of lags.

    K : numpy array
    -   ??? Matrix of innovations-form state space VAR model parameters with
        dimensions [m x n].

    V : numpy array
    -   Covariance matrix of the innovations-form state space VAR model with
        dimensions [n x n].

    freqs : list[int | float]
    -   Frequencies of connectivity being analysed.

    seeds : list[int]
    -   Seed indices. Cannot contain indices also in 'targets'.

    targets : list[int]
    -   Target indices. Cannot contain indices also in 'seeds'.

    RETURNS
    -------
    gc_vals : numpy array
    -   Spectral Granger causality from the seeds to the targets for each
        frequency.

    NOTES
    -----
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Stefan Haufe's research group.
    """
    gc_vals = np.zeros(len(freqs))
    z = np.exp(-1j * np.pi * np.linspace(0, 1, len(freqs)))
    H = ss_params_to_tf(A, C, K, z)
    VSQRT = np.linalg.cholesky(V)
    PVSQRT = np.linalg.cholesky(partial_covariance(V, seeds, targets))

    for freq_i in range(len(freqs)):
        HV = np.matmul(H[:, :, freq_i], VSQRT)
        S = np.matmul(HV, HV.conj().T)
        S11 = S[np.ix_(targets, targets)]
        if len(PVSQRT) == 1:
            HV12 = H[targets, seeds, freq_i] * PVSQRT
            HV12_by_HV12 = np.outer(HV12, HV12.conj().T)
        else:
            HV12 = np.matmul(H[np.ix_(targets, seeds)][:, :, freq_i], PVSQRT)
            HV12_by_HV12 = np.matmul(HV12, HV12.conj().T)
        if len(targets) == 1:
            det_S11 = np.real(S11)
            det_S11_HV12 = np.real(S11 - HV12_by_HV12)
        else:
            det_S11 = np.real(np.linalg.det(S11))
            det_S11_HV12 = np.real(np.linalg.det(S11 - HV12_by_HV12))
        gc_vals[freq_i] = np.log(det_S11) - np.log(det_S11_HV12)

    return gc_vals


def ss_params_to_tf(A: NDArray, C: NDArray, K: NDArray, z: NDArray) -> NDArray:
    """Computes a transfer function (moving-average representation) for
    innovations-form state-space VAR model parameters.

    PARAMETERS
    ----------
    A : numpy array
    -   Matrix of innovations-form state space VAR model parameters with
        dimensions [m x m].

    C : numpy array
    -   Matrix of innovations-form state space VAR model parameters with
        dimensions [n x m].

    K : numpy array
    -   Matrix of innovations-form state space VAR model parameters with
        dimensions [m x n].

    z : numpy array
    -   Vector of points on a unit circle in the complex plane, with length p.

    RETURNS
    -------
    H : numpy array
    -   The transfer function with dimensions [n x n x p]

    RAISES
    ------
    ValueError
    -   If 'A', 'C', or 'K' are not two-dimensional matrices.
    -   If 'z' is not a vector.

    NOTES
    -----
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Stefan Haufe's research group.
    """
    if len(A.shape) != 2:
        raise ValueError(
            f"'A' must be a two-dimensional matrix, but has {len(A.shape)} "
            "dimension(s)."
        )
    if len(C.shape) != 2:
        raise ValueError(
            f"'C' must be a two-dimensional matrix, but has {len(C.shape)} "
            "dimension(s)."
        )
    if len(K.shape) != 2:
        raise ValueError(
            f"'K' must be a two-dimensional matrix, but has {len(K.shape)} "
            "dimension(s)."
        )
    if len(z.shape) != 1:
        raise_err = True
        if len(z.shape) == 2:
            raise_err = False
            if z.shape[1] != 1:
                raise_err = True
        if raise_err:
            raise ValueError("'z' must be a vector, but is a matrix.")

    h = len(z)
    n = C.shape[0]
    m = A.shape[0]
    I_n = np.eye(n)
    I_m = np.eye(m)
    H = np.zeros((n, n, h), dtype=complex)

    for k in range(h):
        H[:, :, k] = I_n + np.matmul(
            C, spla.lu_solve(spla.lu_factor(z[k] * I_m - A), K)
        )

    return H


def partial_covariance(
    V: NDArray, idcs_1: list[int], idcs_2: list[int]
) -> NDArray:
    """Computes the partial covariance for use in spectral Granger causality
    calculations.

    PARAMETERS
    ----------
    V : numpy array
    -   A positive-definite, symmetric covariance matrix.

    idcs_1 : list[int]
    -   First set of indices to use for the partial covariance. Cannot contain
        any values in 'idcs_2'.

    idcs_2 : list[int]
    -   Second set of indices to use for the partial covariance. Cannot contain
        any values in 'idcs_1'.

    RETURNS
    -------
    numpy array
    -   The partial covariance matrix.

    RAISES
    ------
    ValueError
    -   Raised if 'V' is not a two-dimensional matrix.
    -   Raised if 'V' is not a symmetric, positive-definite matrix.
    -   Raised if 'idcs_1' and 'idcs_2' contain common indices.

    NOTES
    -----
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Stefan Haufe's research group.
    """
    if len(V.shape) != 2:
        raise ValueError(
            f"'V' must be a two-dimensional matrix, but has {len(V.shape)} "
            "dimension(s)."
        )
    if not check_posdef(V):
        raise ValueError(
            "'V' must be a positive-definite, symmetric matrix, but it is not."
        )
    common_idcs = set.intersection(set(idcs_1), set(idcs_2))
    if common_idcs:
        raise ValueError(
            "There are common indices present in both sets of indices, but "
            f"this is not allowed.\n- Common indices: {common_idcs}"
        )

    if len(idcs_2) == 1:
        W = (1 / np.sqrt(V[idcs_2, idcs_2])) * V[idcs_2, idcs_1]
        W_by_W = np.outer(W.conj(), W)
    else:
        W = np.linalg.solve(
            np.linalg.cholesky(V[np.ix_(idcs_2, idcs_2)]),
            V[np.ix_(idcs_2, idcs_1)],
        )
        W_by_W = W.conj().T.dot(W)

    return V[np.ix_(idcs_1, idcs_1)] - W_by_W


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
    data_shape = data.shape
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


def discrete_lyapunov(A: NDArray, Q: NDArray) -> NDArray:
    """Solves the discrete-time Lyapunov equation via Schur decomposition with a
    column-by-column solution.

    PARAMETERS
    ----------
    A : numpy array
    -   A square matrix with a spectral radius of < 1.

    Q : numpy array
    -   A symmetric, positive-definite matrix.

    RETURNS
    -------
    X : numpy array
    -   The solution of the discrete-time Lyapunov equation.

    NOTES
    -----
    -   The Lyapunov equation takes the form X = A*X*conj(A)'+Q
    -   References: Kitagawa G., 1977, International Journal of Control, DOI:
        10.1080/00207177708922266; Hammarling S.J., 1982, IMA Journal of
        Numerical Analysis, DOI: 10.1093/imanum/2.3.303.
    -   Translated into Python from the MATLAB MVGC toolbox v1.0 functions
        'dlyap' and 'lyapslv' by Thomas Samuel Binns (@tsbinns).
    """
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError(
            f"The matrix A is not square (has dimensions {A.shape})."
        )
    if A.shape != Q.shape:
        raise ValueError(
            f"The dimensions of matrix Q ({Q.shape}) do not match the "
            f"dimensions of matrix A ({A.shape})."
        )

    T, U = spla.schur(A)
    Q = np.matmul(np.matmul(-U.conj().T, Q), U)

    # Solve the equation column-by-column
    X = np.zeros((n, n))
    j = n - 1
    while j > 0:
        j1 = j + 1

        # Check Schur block size
        if T[j, j - 1] != 0:
            bsiz = 2
            j = j - 1
        else:
            bsiz = 1
        bsizn = bsiz * n

        Ajj = kron(T[j:j1, j:j1], T) - np.eye(bsizn)
        rhs = reshape(Q[:, j:j1], (bsizn, 1))

        if j1 < n:
            rhs = rhs + reshape(
                np.matmul(
                    T,
                    np.matmul(X[:, j1:n], T[j:j1, j1:n].conj().T),
                ),
                (bsizn, 1),
            )

        v = spla.lu_solve(spla.lu_factor(-Ajj), rhs)
        X[:, j] = v[:n].flatten()

        if bsiz == 2:
            X[:, j1 - 1] = v[n:bsizn].flatten()

        j = j - 1

    # Convert back to original coordinates
    X = np.matmul(U, np.matmul(X, U.conj().T))

    return X


def multivariate_connectivity_compute_e(data: NDArray, n_seeds: int) -> NDArray:
    """Computes 'E' as the imaginary part of the transformed connectivity matrix
    'D' derived from the original connectivity matrix 'C' between the seed and
    target signals.
    -   Designed for use with the methods 'max_imaginary_coherence' and
        'multivariate_interaction_measure'.

    data : numpy array
    -   Coherency values between all possible connections of seeds and targets,
        for a single frequency. Has the dimensions [signals x signals].

    n_seeds : int
    -   Number of seed signals. Entries in both dimensions of 'data' from
        '0 : n_seeds' are taken as the coherency values for seed signals.
        Entries from 'n_seeds : end' are taken as the coherency values for
        target signals.

    RETURNS
    -------
    E : numpy array
    -   The imaginary part of the transformed connectivity matrix 'D' between
        seed and target signals.

    NOTES
    -----
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Franziska Pellegrini of Stefan Haufe's research group.
    """
    # Equation 2
    C_aa = data[0:n_seeds, 0:n_seeds]
    C_ab = data[0:n_seeds, n_seeds:]
    C_bb = data[n_seeds:, n_seeds:]
    C_ba = data[n_seeds:, 0:n_seeds]
    C = np.vstack((np.hstack((C_aa, C_ab)), np.hstack((C_ba, C_bb))))

    # Equation 3
    T = np.zeros(C.shape)
    T[0:n_seeds, 0:n_seeds] = spla.fractional_matrix_power(np.real(C_aa), -0.5)
    T[n_seeds:, n_seeds:] = spla.fractional_matrix_power(np.real(C_bb), -0.5)

    # Equation 4
    D = np.matmul(T, np.matmul(C, T))

    # 'E' as the imaginary part of 'D' between seeds and targets
    E = np.imag(D[0:n_seeds, n_seeds:])

    return E
