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

from numpy.typing import NDArray
import numpy as np
from scipy.linalg import fractional_matrix_power


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
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084

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
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084

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
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084

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
