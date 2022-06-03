"""Methods for performing MATLAB-equivalent functions in Python.

METHODS
-------
mrdivide
-   Performs a matrix right division, equivalent to the 'mrdivide' function or
    '/' operator in MATLAB.

-   Equivalent to the MATLAB 'reshape' function, whereby the elements from the
    first axis onwards are taken in some order for the reshaping (i.e. from axis
    0 to n).
"""

from numpy.typing import NDArray
import numpy as np


def mrdivide(numerator: NDArray, denominator: NDArray) -> NDArray:
    """Performs a matrix right division, equivalent to the 'mrdivide' function
    or '/' operator in MATLAB.

    PARAMETERS
    ----------
    numerator : numpy array
    -   An n x m matrix that will be the numerator in the division.

    denominator : numpy array
    -   An n x m matrix that will be the denominator in the division.

    RETURNS
    -------
    numpy array
    -   The result of the matrix right division.
    """

    return np.linalg.solve(denominator.conj().T, numerator.conj().T).conj().T


def reshape(array: NDArray, dims: tuple[int]) -> NDArray:
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

    dims : tuple[int]
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
