"""
This is a processor that can denoise a waveform using a haar_wavelet filter
"""

from __future__ import annotations

import numpy as np
from numba import guvectorize, jit

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


# defined dot product
@jit(nopython=True)
def dot(a, b):
    val = 0
    for i in range(0, len(a)):
        val += a[i] * b[i]

    return val


# defined concatenate two numpy arrays
@jit(nopython=True)
def concat(a, b):
    mat_new = np.zeros((len(a) + len(b), len(a[0])))
    for i in range(0, len(a)):
        mat_new[i] = a[i]
    for i in range(0, len(b)):
        mat_new[i + len(a)] = b[i]
    return mat_new


# defined kronecker product
@jit(nopython=True)
def kron(a, b):
    mat = np.zeros((len(a[0]), len(a) * len(b)))
    idx = 0
    for i in range(0, len(a)):
        for j in range(0, len(a[0])):
            for k in range(0, len(b)):
                mat[i][idx] = a[i][j] * b[k]
                idx += 1

    return mat


# haar wavelet matrix
@jit(nopython=True)
def haar_mat(level):
    s = np.sqrt(2) / 2
    for n in range(0, level + 1):
        if n == 0:
            mat = np.asarray([[1, 1], [1, -1]]) * s
        elif n != 0:
            mat = concat(kron(mat, [1, 1]), kron(np.identity(2**n), [1, -1])) * s

    return mat


# performs the haar standing wavelet transform
@jit(nopython=True)
def haar_swt(seq, level):
    mat = haar_mat(level)
    cds = np.zeros((2 ** (level + 1), len(seq)))
    for k in range(0, len(cds)):
        for i in range(0, len(seq)):
            sub_seq = seq[i : i + 2 ** (level + 1)]
            if len(sub_seq) != len(mat[k]) and i < len(seq) - 1:
                sub_seq = np.asarray(
                    [
                        seq[j]
                        for j in range(i - len(seq), i - len(seq) + 2 ** (level + 1), 1)
                    ]
                )
            elif i == len(seq) - 1:
                sub_seq = np.asarray(
                    [seq[j] for j in range(-1, -1 + 2 ** (level + 1), 1)]
                )

            cds[k][i] = dot(mat[k], sub_seq)

    return cds


# performs the inverse wavelet transform
@jit(nopython=True)
def haar_iswt(cds: np.ndarray, level: np.int64) -> np.ndarray:
    seq = np.zeros(len(cds[0]))
    mat = haar_mat(level=level)
    mat_t = np.transpose(mat)
    for i in range(0, len(seq)):
        sub_seq = np.asarray([cds[j][i] for j in range(0, len(cds))])
        seq[i] = dot(mat_t[0], sub_seq)
    return seq


# performs the actual filter
@jit(nopython=True)
def cut_wave(cds: np.ndarray, level: np.int64) -> np.ndarray:
    threshold = np.zeros(2 ** (level + 1))
    for i in range(1, len(cds)):
        median_value = np.median(cds[i])
        median_average_deviation = np.median(np.absolute(cds[i] - median_value))
        sig1 = median_average_deviation / 0.6745
        threshold[i] = 1.2 * np.float64(sig1 * np.sqrt(2 * np.log(len(cds[i]))))
    # threshold = np.float64(np.sqrt(2*np.log(len(wave))))

    for idx in range(1, len(cds)):
        for i in range(0, len(cds[idx])):
            if np.absolute(cds[idx][i]) < threshold[idx]:
                cds[idx][i] = np.float64(0.0)

    return cds


# the actual function that is called in dsp
@guvectorize(["void(float64[:], int64, float64[:])"], "(n),()->(n)", **nb_kwargs)
def denoise_wave(wave: np.ndarray, level: np.int64, wave_denoise: np.ndarray) -> None:
    """Apply a symmetric trapezoidal filter to the waveform.

    Parameters
    ----------
    wave
        the input waveform.
    level
        the level at which you filter the waveform.
    wave_denoise
        the filtered waveform.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "wave_denoise": {
            "function": "denoise_wave",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "7", "wave_denoise"],
            "unit": "ADC"
        }
    """

    cds = haar_swt(wave, level)
    cds = cut_wave(cds, level)
    wave = haar_iswt(cds, level)
    wave_denoise[:] = wave[:]
