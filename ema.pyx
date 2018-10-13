cimport cython
cimport numpy as np
import numpy as np

def untyped_ema(xs, smooth, init):
    smoothed = np.empty_like(xs)
    for t in range(len(xs)):
        if t == 0:
            smoothed[t] = init
        else:
            smoothed[t] = (1-smooth)*xs[t-1] + smooth*smoothed[t-1]
    return smoothed

@cython.boundscheck(False)
@cython.wraparound(False)
def typed_ema(
        np.ndarray[np.float64_t, ndim=1] xs,
        double smooth,
        double init):
    cdef np.ndarray[np.float64_t, ndim=1] smoothed = (
        np.empty_like(xs, dtype=np.float64)
    )
    cdef int t
    cdef int length = len(xs)
    with cython.nogil:
        for t in range(length):
            if t == 0:
                smoothed[t] = init
            else:
                smoothed[t] = (1-smooth)*xs[t-1] + smooth*smoothed[t-1]
    return smoothed
