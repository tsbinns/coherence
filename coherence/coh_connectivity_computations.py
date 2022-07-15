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
import numpy as np
from numpy.typing import NDArray
from coh_matlab_functions import reshape
from coh_signal_processing_computations import (
    autocov_to_full_var,
    csd_to_autocov,
    full_var_to_iss,
    iss_to_usgc,
    mim_mic_compute_e,
    mim_mic_cross_spectra_svd,
)

from scipy.io import loadmat


def multivariate_interaction_measure(
    C: NDArray,
    n_seeds: int,
    n_targets: int,
    n_seed_components: Union[int, None] = None,
    n_target_components: Union[int, None] = None,
) -> NDArray:
    """Computes the multivariate interaction measure between two groups of
    signals.

    C : numpy array
    -   Coherency values between all possible connections of two groups of
        signals, A and B, across frequencies. Has the dimensions [signals x
        signals x frequencies].

    n_seeds : int
    -   Number of seed signals. Entries in the first two dimensions of 'C' from
        [0 : n_seeds] are taken as the coherency values for seed signals.

    n_targets : int
    -   Number of target signals. Entries in the first two dimensions of 'C'
        from ['n_seeds : n_targets'] are taken as the coherency values for
        target signals.

    n_seed_components : int | None; default None
    -   The number of components that should be taken from the seed data after
        dimensionality reduction using singular value decomposition (SVD).
    -   If 'None', no dimensionality reduction is performed on the seed data.

    n_target_components : int | None; default None
    -   The number of components that should be taken from the target data after
        dimensionality reduction using SVD.
    -   If 'None', no dimensionality reduction is performed on the target data.

    RETURNS
    -------
    mim : numpy array
    -   One-dimensional array containing a connectivity value between seeds and
        targets for each frequency.

    RAISES
    ------
    ValueError
    -   Raised if the data is not a three-dimensional array.
    -   Raised if the first two dimensions of 'data' is not a square matrix with
        lengths equal to the combined number of seed and target signals.

    NOTES
    -----
    -   References: [1] Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Franziska Pellegrini of Stefan Haufe's research group.
    """
    if len(C.shape) != 3:
        raise ValueError(
            "Error when computing MIC:\nThe data must be a three-dimensional "
            "array containing connectivity values across frequencies, but the "
            f"data has {len(C.shape)} dimensions."
        )
    n_signals = n_seeds + n_targets
    if (n_signals, n_signals) != C.shape[0:2]:
        raise ValueError(
            "Error when calculating the multivariate interaction measure:\nThe "
            f"data for each frequency must be a [{n_signals} x {n_signals}] "
            "square matrix containing all connectivities between the "
            f"{n_seeds} seed and {n_targets} target signals in, but it is a "
            f"[{np.shape(C)[0]} x {np.shape(C)[1]}] matrix."
        )

    n_freqs = C.shape[2]
    mim = np.empty(n_freqs)
    for freq_i in range(n_freqs):
        # Eqs. 32 & 33
        C_bar, U_bar_aa, _ = mim_mic_cross_spectra_svd(
            C=C[:, :, freq_i],
            n_seeds=n_seeds,
            n_seed_components=n_seed_components,
            n_target_components=n_target_components,
        )

        # Eqs. 3 & 4
        E = mim_mic_compute_e(C=C_bar, n_seeds=U_bar_aa.shape[1])

        # Equation 14
        mim[freq_i] = np.trace(np.matmul(E, np.conj(E).T))

    return mim


def max_imaginary_coherence(
    C: NDArray,
    n_seeds: int,
    n_targets: int,
    n_seed_components: Union[int, None] = None,
    n_target_components: Union[int, None] = None,
    return_topographies: bool = True,
) -> Union[NDArray, list[NDArray]]:
    """Computes the maximised imaginary coherence between two groups of signals.

    C : numpy array
    -   Coherency values between all possible connections of two groups of
        signals, A and B, across frequencies. Has the dimensions [signals x
        signals x frequencies].

    n_seeds : int
    -   Number of seed signals. Entries in the first two dimensions of 'C' from
        [0 : n_seeds] are taken as the coherency values for seed signals.

    n_targets : int
    -   Number of target signals. Entries in the first two dimensions of 'C'
        from [n_seeds : n_targets] are taken as the coherency values for target
        signals.

    n_seed_components : int | None; default None
    -   The number of components that should be taken from the seed data after
        dimensionality reduction using singular value decomposition (SVD).
    -   If 'None', no dimensionality reduction is performed on the seed data.

    n_target_components : int | None; default None
    -   The number of components that should be taken from the target data after
        dimensionality reduction using SVD.
    -   If 'None', no dimensionality reduction is performed on the target data.

    return_topographies : bool; default True
    -   Whether or not to return spatial topographies of connectivity for the
        signals.

    RETURNS
    -------
    mic : numpy array
    -   One-dimensional array containing a connectivity value between seed and
        target signals for each frequency.

    tuple(numpy array)
    -   Spatial topographies of connectivity for the seed and target signals,
        respectively, for each frequency, each with dimensions [signals x
        frequencies].
    -   Returned only if 'return_topographies' is 'True'.

    RAISES
    ------
    ValueError
    -   Raised if the data is not a three-dimensional array.
    -   Raised if the first two dimensions of 'data' is not a square matrix with
        lengths equal to the combined number of seed and target signals.

    NOTES
    -----
    -   References: [1] Ewald et al. (2012), NeuroImage. DOI:
        10.1016/j.neuroimage.2011.11.084; [2] Nikulin et al. (2011), NeuroImage,
        DOI: 10.1016/j.neuroimage.2011.01.057.
    -   Spatial topographies are computed using the weight vectors alpha and
        beta (see [1]) by multiplying the real part of the coherency
        cross-spectrum 'C' by weight vectors, as in Eq. 20 of [2], with the
        absolute values then taken. If dimensionality reduction is performed,
        weight vectors are recovered in the original sensor space using Eqs. 46
        & 47 of [1].
    -   Translated into Python by Thomas Samuel Binns (@tsbinns) from MATLAB
        code provided by Franziska Pellegrini of Stefan Haufe's research group.
    """
    if len(C.shape) != 3:
        raise ValueError(
            "Error when computing MIC:\nThe data must be a three-dimensional "
            "array containing connectivity values across frequencies, but the "
            f"data has {len(C.shape)} dimensions."
        )
    n_signals = n_seeds + n_targets
    if (n_signals, n_signals) != C.shape[0:2]:
        raise ValueError(
            "Error when calculating the multivariate interaction measure:\nThe "
            f"data for each frequency must be a [{n_signals} x {n_signals}] "
            "square matrix containing all connectivities between the "
            f"{n_seeds} signals in group A and the {n_targets} signals in "
            f"group B, but it is a [{C.shape[0]} x {C.shape[1]}] "
            "matrix."
        )

    n_freqs = C.shape[2]
    mic = np.empty(n_freqs)
    topos_a = []
    topos_b = []
    for freq_i in range(n_freqs):
        # Eqs. 32 & 33
        C_bar, U_bar_aa, U_bar_bb = mim_mic_cross_spectra_svd(
            C=C[:, :, freq_i],
            n_seeds=n_seeds,
            n_seed_components=n_seed_components,
            n_target_components=n_target_components,
        )

        # Eqs. 3 & 4
        E = mim_mic_compute_e(C=C_bar, n_seeds=U_bar_aa.shape[1])

        # Weights for signals in the groups
        w_a, V_a = np.linalg.eig(np.matmul(E, np.conj(E).T))
        w_b, V_b = np.linalg.eig(np.matmul(np.conj(E).T, E))
        alpha = V_a[:, w_a.argmax()]
        beta = V_b[:, w_b.argmax()]

        # Eq. 7
        mic[freq_i] = np.abs(
            np.matmul(np.conj(alpha).T, np.matmul(E, beta))
            / np.linalg.norm(alpha)
            * np.linalg.norm(beta)
        )

        # Eqs. 46 & 47
        if return_topographies:
            topos_a.append(
                np.abs(
                    np.matmul(
                        np.real(C[:n_seeds, :n_seeds, freq_i]),
                        U_bar_aa.dot(alpha),
                    )
                )
            )  # C_aa * U_bar_aa * alpha
            topos_b.append(
                np.abs(
                    np.matmul(
                        np.real(C[n_seeds:, n_seeds:, freq_i]),
                        U_bar_bb.dot(beta),
                    )
                )
            )  # C_bb * U_bar_bb * beta

    if return_topographies:
        topos_a = np.transpose(topos_a, (1, 0))
        topos_b = np.transpose(topos_b, (1, 0))
        return mic, (topos_a, topos_b)
    else:
        return mic


def granger_causality(
    csd: NDArray,
    freqs: list[Union[int, float]],
    method: str,
    seeds: list[list[int]],
    targets: list[list[int]],
    n_lags: int = 20,
) -> NDArray:
    """Computes frequency-domain Granger causality (GC) between each set of
    seeds and targets.

    PAREMETERS
    ----------
    csd : numpy array
    -   Matrix containing the cross-spectral density between signals, with
        dimensions [n_signals x n_signals x n_frequencies].

    freqs : list[int | float]
    -   Frequencies in 'csd'.

    method : str
    -   Which form of GC to compute.
    -   Supported inputs are: "gc" for GC from seeds to targets; "net_gc" for
        net GC, i.e. GC from seeds to targets minus GC from targets to seeds;
        "trgc" for time-reversed GC (TRGC) from seeds to targets; and "net_trgc"
        for net TRGC, i.e. TRGC from seeds to targets minus TRGC from targets to
        seeds.

    seeds : list[list[int]]
    -   Indices of signals in 'csd' to treat as seeds. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of seeds.
    -   The number of sublists must match the number of sublists in 'targets'.

    targets : list[list[int]]
    -   Indices of signals in 'csd' to treat as targets. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of targets.
    -   The number of sublists must match the number of sublists in 'seeds'.

    n_lags : int; default 20
    -   Number of lags to use when computing the autocovariance sequence from
        the cross-spectra.

    RETURNS
    -------
    numpy array
    -   Granger causality values in a matrix with dimensions [n_nodes x
        n_frequencies], where the nodes correspond to seed-target pairs.

    NOTES
    -----
    -   Net TRGC is the recommended method for maximum robustness.
    -   Each group of seeds and targets cannot contain the same indices.
    """
    autocov = csd_to_autocov(csd, n_lags)

    return gc_computation(
        autocov=autocov,
        freqs=freqs,
        method=method,
        seeds=seeds,
        targets=targets,
    )


def gc_computation(
    autocov: NDArray,
    freqs: list[Union[int, float]],
    method: str,
    seeds: list[list[Union[int, float]]],
    targets: list[list[Union[int, float]]],
) -> NDArray:
    """Computes frequency-domain Granger causality.

    PARAMETERS
    ----------
    autocov : numpy array
    -   An autocovariance sequence with dimensions [n_signals x n_signals x
        n_lags + 1].

    freqs : list[int | float]
    -   Frequencies of the data being analysed.

    method : str
    -   Which form of GC to compute.
    -   Supported inputs are: "gc" for GC from seeds to targets; "net_gc" for
        net GC, i.e. GC from seeds to targets minus GC from targets to seeds;
        "trgc" for time-reversed GC (TRGC) from seeds to targets; and "net_trgc"
        for net TRGC, i.e. TRGC from seeds to targets minus TRGC from targets to
        seeds.

    seeds : list[list[int]]
    -   Indices of signals in 'csd' to treat as seeds. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of seeds.
    -   The number of sublists must match the number of sublists in 'targets'.

    targets : list[list[int]]
    -   Indices of signals in 'csd' to treat as targets. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of targets.
    -   The number of sublists must match the number of sublists in 'seeds'.

    RETURNS
    -------
    gc_vals : numpy array
    -   Granger causality values in a matrix with dimensions [n_nodes x
        n_frequencies], where the nodes correspond to seed-target pairs.

    RAISES
    ------
    NotImplementedError
    -   Raised if 'method' is not supported.
    ValueError
    -   Raised if the number of seed and target groups do not match.
    """
    supported_methods = ["gc", "net_gc", "trgc", "net_trgc"]
    if method not in supported_methods:
        raise NotImplementedError(
            f"The method '{method}' for computing frequency-domain Granger "
            "causality is not recognised. Supported methods are "
            f"{supported_methods}."
        )
    if len(seeds) != len(targets):
        raise ValueError(
            f"The number of seed and target groups ({len(seeds)} and "
            f"{len(targets)}, respectively) do not match."
        )

    gc_vals = np.zeros((len(seeds), len(freqs)))
    node_i = 0
    for seed_idcs, target_idcs in zip(seeds, targets):
        node_idcs = [*seed_idcs, *target_idcs]
        seed_idcs_new = np.arange(len(seed_idcs)).tolist()
        target_idcs_new = np.arange(
            start=len(seed_idcs), stop=len(node_idcs)
        ).tolist()
        var_coeffs, residuals_cov = autocov_to_full_var(
            autocov[np.ix_(node_idcs, node_idcs, np.arange(autocov.shape[2]))],
            enforce_posdef_residuals_cov=True,
        )
        var_coeffs_2d = reshape(
            var_coeffs,
            (var_coeffs.shape[0], var_coeffs.shape[0] * var_coeffs.shape[2]),
        )
        A, K = full_var_to_iss(AF=var_coeffs_2d, V=residuals_cov)
        gc_vals[node_i, :] = iss_to_usgc(
            A=A,
            C=var_coeffs_2d,
            K=K,
            V=residuals_cov,
            freqs=freqs,
            seeds=seed_idcs_new,
            targets=target_idcs_new,
        )
        if "net" in method:
            gc_vals[node_i, :] -= iss_to_usgc(
                A=A,
                C=var_coeffs_2d,
                K=K,
                V=residuals_cov,
                freqs=freqs,
                seeds=target_idcs_new,
                targets=seed_idcs_new,
            )
        node_i += 1

    if "trgc" in method:
        if method == "trgc":
            tr_method = "gc"
        elif method == "net_trgc":
            tr_method = "net_gc"
        gc_vals -= gc_computation(
            autocov=np.transpose(autocov, (1, 0, 2)),
            freqs=freqs,
            method=tr_method,
            seeds=seeds,
            targets=targets,
        )

    return gc_vals


"""
csd = loadmat("coherence\\csd.mat")["CS"]
gc_vals = granger_causality(
    csd=csd,
    freqs=np.arange(csd.shape[2]),
    method="trgc",
    seeds=[[0], [1], [2]],
    targets=[[3, 4, 5, 6] for n in range(3)],
    n_lags=20,
)
print("jeff")
"""
