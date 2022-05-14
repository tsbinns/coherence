"""Methods for computing connectivity metrics.

METHODS
-------
multivariate_interaction_measure
-   Computes the multivariate interaction measure between two groups of signals.

max_imaginary_coherence
-   Computes the maximised imaginary coherence between two groups of signals.
"""

from numpy.typing import NDArray
import numpy as np
from scipy.io import loadmat
from scipy.linalg import fractional_matrix_power

signal = loadmat("C:\\Users\\User\\GitHub\\coherence\\coherence\\test_data.mat")
signal = np.asarray(signal["COH"])


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

    RAISES
    ------
    ValueError
    -   Raised if the first two dimensions of 'data' is not a square matrix with
        lengths equal to the combined number of signals in groups A and B.
    """

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
        # Equation 2
        C_aa = data[0:n_group_a, 0:n_group_a, freq_i]
        C_ab = data[0:n_group_a, n_group_a:, freq_i]
        C_bb = data[n_group_a:, n_group_a:, freq_i]
        C_ba = data[n_group_a:, 0:n_group_a, freq_i]
        C = np.vstack((np.hstack((C_aa, C_ab)), np.hstack((C_ba, C_bb))))

        # Equation 3
        T = np.zeros(np.shape(C))
        T[0:n_group_a, 0:n_group_a] = fractional_matrix_power(
            np.real(C_aa), -0.5
        )
        T[n_group_a:, n_group_a:] = fractional_matrix_power(np.real(C_bb), -0.5)

        # Equation 4
        D = np.matmul(T, np.matmul(C, T))

        # 'E' as the imaginary part of 'D' between groups A and B
        E = np.imag(D[0:n_group_a, n_group_a:])

        # Equation 14
        mim[freq_i] = np.trace(np.matmul(E, E.T))

    return mim


def max_imaginary_coherence(data: NDArray, data_b: NDArray) -> NDArray:
    """Computes the maximised imaginary coherence between two groups of signals.
    -   Follows the approach set out in Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084
    """


mim = multivariate_interaction_measure(data=signal, n_group_a=6, n_group_b=8)
