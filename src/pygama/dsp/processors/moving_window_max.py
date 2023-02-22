"""
This processor is used to calculate multiple energies
from multiple signals in a waveform.

This processor works best for external trigger waveforms
where you don't know where any signals are in the waveform.
"""

from __future__ import annotations

import numpy as np
from numba import guvectorize, jit

from pygama.dsp.utils import numba_defaults_kwargs as nb_kwargs


# This is an algorithm to find the mode every value in an array
@jit(nopython=True)
def find_modes(arr: np.ndarray) -> np.ndarray:
    energies_rep = np.zeros((2, len(arr)))
    energies_rep[0][0] = arr[0]

    count = 0
    for i in range(0, len(arr)):
        if np.abs(np.int64(arr[0] - arr[i])) <= 0.1:
            count += 1
        else:
            break

    energies_rep[1][0] = count

    for i in range(1, len(arr)):
        count = 0
        if arr[i] == arr[i - 1]:
            energies_rep[0][i] = energies_rep[0][i - 1]
            energies_rep[1][i] = energies_rep[1][i - 1]
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                count += 1
            else:
                break

        energies_rep[0][i] = arr[i]
        energies_rep[1][i] = count

    return energies_rep


# There is a find_idx algorithm to get an onset time
@jit(nopython=True)
def find_idx(arr, val, idx_begin):
    for i in range(idx_begin - 1, 0, -1):
        count = arr[i]
        if np.abs(count - val) <= 0.2:
            break

    idx = i

    return idx


@guvectorize(
    [
        "void(float32[:], float64, float64, float32[:], float32[:])",
        "void(float64[:], float64, float64, float64[:], float64[:])",
    ],
    "(n),(),()->(n),(n)",
    **nb_kwargs,
    nopython=True,
)
def moving_window_max(
    w_trap: np.ndarray,
    rise: np.float64,
    flat: np.float64,
    energies: np.ndarray,
    onset_times: np.ndarray,
) -> np.ndarray:
    """Apply a symmetric trapezoidal filter to the waveform.

    Parameters
    ----------
    wave
        the input waveform.
    rise
        rise value of the trap filter in us.
    flat
        flat value of the trap filter
    energies
        array of energies found. This is array is as long as the waveform so you get a lot of zeros.
    onset_times
        array of energies found. This is array is as long as the waveform so you get a lot of zeros.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "energies, onset_times": {
            "function": "moving_window_max",
            "module": "pygama.dsp.processors",
            "args": ["waveform", "0.7*us", "0.2*us", "energies", "onset_times"],
            "unit": "ADC"
        }
    """
    window = int(2 * (rise) + (flat))
    wind_trap_energy = np.zeros(len(w_trap) - 2 * window)
    wind_trap_time = np.zeros(len(w_trap) - 2 * window)
    j = 0
    for i in range(window, len(w_trap) - window):
        wind_trap_energy[j] = np.max(w_trap[i : window + i])
        wind_trap_time[j] = np.argmax(w_trap[i : window + i]) + i
        j += 1

    energy_lis = find_modes(wind_trap_energy)

    m = 0
    for i in range(1, len(energy_lis[0])):
        if energy_lis[1][i] > flat:
            if energy_lis[0][i] == energy_lis[0][i - 1]:
                continue
            energies[m] = energy_lis[0][i]
            onset_times[m] = wind_trap_time[i]
            m += 1

    for i in range(0, len(onset_times)):
        onset_times[i] = find_idx(w_trap, 0.1 * energies[i], onset_times[i])
