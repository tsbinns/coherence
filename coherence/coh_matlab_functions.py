"""Methods for performing MATLAB-equivalent functions in Python.

METHODS
-------
mrdivide
-   Performs a matrix right division, equivalent to the 'mrdivide' function or
    '/' operator in MATLAB.

mldivide
-   Performs a matrix left division, equivalent to the 'mrdivide' function or \
    operator in MATLAB.

reshape
-   Equivalent to the MATLAB 'reshape' function, whereby the elements from the
    first axis onwards are taken in some order for the reshaping (i.e. from axis
    0 to n).

kron
-   Equivalent to the MATLAB 'kron' function, in which the Kronecker product is
    calculated.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np
import scipy as sp


def mrdivide(A: NDArray, B: NDArray) -> NDArray:
    """Performs a matrix right division, equivalent to the 'mrdivide' function
    or '/' operator in MATLAB.

    PARAMETERS
    ----------
    A : numpy array
    -   An n x m matrix on the left side of the division.

    B : numpy array
    -   An n x m matrix on the right side of the division.

    RETURNS
    -------
    numpy array
    -   The result of the matrix right division.
    """

    return np.linalg.solve(B.conj().T, A.conj().T).conj().T


def mldivide(A: NDArray, B: NDArray) -> NDArray:
    """Performs a matrix left division, equivalent to the 'mldivide' function or
    \ operator in MATLAB.

    PARAMETERS
    ----------
    A : numpy array
    -   An n x m matrix on the left side of the division.

    B : numpy array
    -   An n x m matrix on the right side of the division.

    RETURNS
    -------
    numpy array
    -   The result of the matrix left division.
    """

    return np.linalg.lstsq(A, B, rcond=None)[0]


def reshape(array: NDArray, dims: Union[int, tuple[int]]) -> NDArray:
    """Equivalent to the MATLAB 'reshape' function, whereby the elements from
    the first axis onwards are taken in some order for the reshaping (i.e. from
    axis 0 to n).
    -   This is different to numpy's method of taking elements from the last
        axis first, then the penultimate axis, and so on (i.e. from axis n to
        0).

    PARAMETERS
    ----------
    array : numpy array
    -   Array which will be reshaped.

    dims : int | tuple[int]
    -   The dimensions of the reshaped array.

    RETURNS
    -------
    numpy array
    -   The reshaped array.

    NOTES
    -----
    -   This is equivalent to calling numpy.reshape(array, order="F").
    """

    return np.reshape(array, dims, order="F")


def kron(A: NDArray, B: NDArray) -> NDArray:
    """Equivalent to the MATLAB 'kron' function, in which the Kronecker product
    is calculated.

    PARAMETERS
    ----------
    A : numpy array
    -   A matrix.

    B : numpy array
    -   A matrix.

    RETURNS
    -------
    K : numpy array
    -   The Kronecker product of 'A' and 'B'.

    NOTES
    -----
    -   If the matrices are not sparse, the numpy 'kron' function is used. If
        either of the matrices are sparse, the scipy.sparse function 'kron' is
        used.
    """

    if not sp.sparse.issparse(A) and not sp.sparse.issparse(B):
        K = np.kron(A, B)
    else:
        K = sp.sparse.kron(A, B)

    return K
