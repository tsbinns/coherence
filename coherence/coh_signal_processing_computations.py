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
from coh_matlab_functions import reshape, kron, linsolve_transa

from scipy.io import loadmat


def csd_to_autocov(csd: NDArray, n_lags: Union[int, None] = None) -> NDArray:
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


def autocov_to_full_var(
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
            # n_lags = G.shape[2]
            # if n_lags - 1 > 1:
            #    _, V = autocov_to_full_var(
            #        G=G[:, :, : n_lags - 1],
            #        enforce_posdef_residuals_cov=True,
            #    )
            # else:
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

    # equivalent to calling A/B or linsolve(B',A',opts.TRANSA=true)' in MATLAB
    AF[:, kf] = linsolve_transa(G0.conj().T, GB[kb, :].conj().T).conj().T
    AB[:, kb] = linsolve_transa(G0.conj().T, GF[kf, :].conj().T).conj().T
    # AF[:, kf] = (
    #    spla.solve(G0.conj().T, GB[kb, :].conj().T, transposed=True).conj().T
    # )
    # AB[:, kb] = (
    #    spla.solve(G0.conj().T, GF[kf, :].conj().T, transposed=True).conj().T
    # )

    ### Recursion
    for k in np.arange(2, q + 1):
        # equivalent to calling A/B or linsolve(B,A',opts.TRANSA=true)' in MATLAB
        var_A = GB[(r - 1) * n : r * n, :] - np.matmul(AF[:, kf], GB[kb, :])
        var_B = G0 - np.matmul(AB[:, kb], GB[kb, :])
        AAF = linsolve_transa(var_B, var_A.conj().T).conj().T
        var_A = GF[(k - 1) * n : k * n, :] - np.matmul(AB[:, kb], GF[kf, :])
        var_B = G0 - np.matmul(AF[:, kf], GF[kf, :])
        AAB = linsolve_transa(var_B, var_A.conj().T).conj().T
        # A = GB[(r - 1) * n : r * n, :] - np.matmul(AF[:, kf], GB[kb, :])
        # B = G0 - np.matmul(AB[:, kb], GB[kb, :])
        # AAF = spla.solve(B, A.conj().T, transposed=True).conj().T  # DF/VB
        # A = GF[(k - 1) * n : k * n, :] - np.matmul(AB[:, kb], GF[kf, :])
        # B = G0 - np.matmul(AF[:, kf], GF[kf, :])
        # AAB = spla.solve(B, A.conj().T, transposed=True).conj().T  # DB/VF

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


def full_var_to_iss(AF: NDArray, V: NDArray):
    """Computes innovations-form parameters for a state-space model from a full
    vector autoregressive (VAR) model using Aoki's method.

    For a non-moving-average full VAR model, the state-space parameters C (the
    observation matrix) and V (the innivations covariance matrix) are identical
    to AF and V of the VAR model, respectively.

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
    -   State transition matrix??

    K : numpy array
    -   Kalman gain matix??

    NOTES
    -----
    -   Reference(s): [1] Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   Aoki's method for computing innovations-form parameters for a
        state-space model allows for zero-lag coefficients.
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
    A = np.vstack((AF, Ip[: (len(Ip) - m), :]))  # state transition matrix?
    K = np.vstack(
        (np.eye(m), np.zeros(((m * (p - 1)), m)))
    )  # Kalman gain matrix?

    # O = discrete_lyapunov(A=A, Q=-np.matmul(K, np.matmul(V, K.conj().T)))
    # lambda_0 = (
    #    np.matmul(AF, np.matmul(O, AF.conj().T)) + V
    # )  # variance of the process

    return A, K


def iss_to_usgc(
    A: NDArray,
    C: NDArray,
    K: NDArray,
    V: NDArray,
    freqs: list[Union[int, float]],
    seeds: list[int],
    targets: list[int],
) -> NDArray:
    """Computes unconditional spectral Granger causality from innovations-form
    state-space model parameters.

    PARAMETERS
    ----------
    A : numpy array
    -   State transition matrix?? with dimensions [m x m].

    C : numpy array
    -   Observation matrix?? with dimensions [n x m], where 'n' is the number of
        signals and 'm' the number of signals times the number of lags.

    K : numpy array
    -   Kalman gain matrix?? with dimensions [m x n].

    V : numpy array
    -   Innovations covariance matrix?? with dimensions [n x n].

    freqs : list[int | float]
    -   Frequencies of connectivity being analysed.

    seeds : list[int]
    -   Seed indices. Cannot contain indices also in 'targets'.

    targets : list[int]
    -   Target indices. Cannot contain indices also in 'seeds'.

    RETURNS
    -------
    f : numpy array
    -   Spectral Granger causality from the seeds to the targets for each
        frequency.

    NOTES
    -----
    -   Reference(s): [1] Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Stefan Haufe's research group.
    """
    f = np.zeros(len(freqs))
    z = np.exp(
        -1j * np.pi * np.linspace(0, 1, len(freqs))
    )  # points on a unit circle in the complex plane, one for each frequency
    H = iss_to_tf(A, C, K, z)  # spectral transfer function
    VSQRT = np.linalg.cholesky(V)
    PVSQRT = np.linalg.cholesky(partial_covariance(V, seeds, targets))

    for freq_i in range(len(freqs)):
        HV = np.matmul(H[:, :, freq_i], VSQRT)
        S = np.matmul(
            HV, HV.conj().T
        )  # CSD of the projected state variable (Eq. 6 of [1])
        S_tt = S[np.ix_(targets, targets)]  # CSD between targets
        if len(PVSQRT) == 1:
            HV_ts = H[targets, seeds, freq_i] * PVSQRT
            HVH_ts = np.outer(HV_ts, HV_ts.conj().T)
        else:
            HV_ts = np.matmul(H[np.ix_(targets, seeds)][:, :, freq_i], PVSQRT)
            HVH_ts = np.matmul(HV_ts, HV_ts.conj().T)
        if len(targets) == 1:
            numerator = np.real(S_tt)
            denominator = np.real(S_tt - HVH_ts)
        else:
            numerator = np.real(np.linalg.det(S_tt))
            denominator = np.real(np.linalg.det(S_tt - HVH_ts))
        f[freq_i] = np.log(numerator) - np.log(denominator)  # Eq. 11 of [1]

    return f


def iss_to_tf(A: NDArray, C: NDArray, K: NDArray, z: NDArray) -> NDArray:
    """Computes a transfer function (moving-average representation) for
    innovations-form state-space model parameters.

    PARAMETERS
    ----------
    A : numpy array
    -   State transition matrix?? with dimensions [m x m].

    C : numpy array
    -   Observation matrix?? with dimensions [n x m].

    K : numpy array
    -   Kalman gain matrix?? with dimensions [m x n].

    z : numpy array
    -   The back-shift operator with length p.

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
    -   Reference: [1] Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   In the frequency domain, the back-shift operator, z, is a vector of
        points on a unit circle in the complex plane. z = e^-iw, where -pi < w
        <= pi. See [17] of [1].
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
            C, spla.lu_solve(spla.lu_factor(z[k] * I_m - A), K)  # Eq. 4 of [1]
        )

    return H


def partial_covariance(
    V: NDArray, seeds: list[int], targets: list[int]
) -> NDArray:
    """Computes the partial covariance for use in spectral Granger causality
    (GC) calculations.

    PARAMETERS
    ----------
    V : numpy array
    -   A positive-definite, symmetric innovations covariance matrix.

    seeds : list[int]
    -   Indices of entries in 'V' that are seeds in the GC calculation.

    targets : list[int]
    -   Indices of entries in 'V' that are targets in the GC calculation.

    RETURNS
    -------
    numpy array
    -   The partial covariance matrix between the targets given the seeds.

    RAISES
    ------
    ValueError
    -   Raised if 'V' is not a two-dimensional matrix.
    -   Raised if 'V' is not a symmetric, positive-definite matrix.
    -   Raised if 'seeds' and 'targets' contain common indices.

    NOTES
    -----
    -   Reference: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   Given a covariance matrix V, the partial covariance matrix of V between
        indices i and j, given k (V_ij|k), is equivalent to
        V_ij - V_ik * V_kk^-1 * V_kj. In this case, i and j are seeds, and k is
        the targets.
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
    common_idcs = set.intersection(set(seeds), set(targets))
    if common_idcs:
        raise ValueError(
            "There are common indices present in both sets of indices, but "
            f"this is not allowed.\n- Common indices: {common_idcs}"
        )

    if len(targets) == 1:
        W = (1 / np.sqrt(V[targets, targets])) * V[targets, seeds]
        W = np.outer(W.conj().T, W)
    else:
        W = np.linalg.solve(
            np.linalg.cholesky(V[np.ix_(targets, targets)]),
            V[np.ix_(targets, seeds)],
        )
        W = W.conj().T.dot(W)

    return V[np.ix_(seeds, seeds)] - W


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


# G = loadmat("G_rand.mat")["G"]
# AF, V = autocovariance_to_full_var(G, enforce_posdef_residuals_cov=True)
# print("jeff")
