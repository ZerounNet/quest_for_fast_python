cimport cython
cimport numpy as np
import numpy as np
from cytoolz import accumulate

def untyped_ema(xs, smooth, init):
    smoothed = np.empty_like(xs)
    for t in range(len(xs)):
        if t == 0:
            smoothed[t] = init
        else:
            smoothed[t] = (1-smooth)*xs[t-1] + smooth*smoothed[t-1]
    return smoothed

def typed_ema(
        np.ndarray[np.float64_t, ndim=1] xs,
        double smooth,
        double init):
    cdef np.ndarray[np.float64_t, ndim=1] smoothed = (
        np.empty_like(xs, dtype=np.float64)
    )
    cdef int t
    cdef int length = len(xs)
    for t in range(length):
        if t == 0:
            smoothed[t] = init
        else:
            smoothed[t] = (
                (1-smooth)*xs[t-1] +
                smooth*smoothed[t-1]
            )
    return smoothed

cdef ema_step(last_smoothed: double, next_value: double, smooth: double):
    return (1-smooth)*next_value + smooth*last_smoothed

def reduce_ema(
        np.ndarray[np.float64_t, ndim=1] xs,
        double smooth,
        double init):
    smoothed = accumulate(lambda s, x: ema_step(s, x, smooth), xs[:-1], init)
    return np.fromiter(smoothed, dtype=np.float64)
