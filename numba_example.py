import numpy as np
from numba import jit

def vanilla_ema(xs, smooth, init):
    smoothed = np.empty_like(xs)
    for t in range(len(xs)):
        if t == 0:
            smoothed[t] = init
        else:
            smoothed[t] = (1-smooth)*xs[t-1] + smooth*smoothed[t-1]
    return smoothed

@jit
def jitted_ema(xs, smooth, init):
    smoothed = np.empty_like(xs)
    for t in range(len(xs)):
        if t == 0:
            smoothed[t] = init
        else:
            smoothed[t] = (
                (1-smooth)*xs[t-1] +
                smooth*smoothed[t-1]
            )
    return smoothed


'''
import pyximport; pyximport.install()
from ema import untyped_ema, typed_ema

xs = np.random.randn(10000)
%timeit _ = vanilla_ema(xs, smooth=0.8, init=0.0)
%timeit _ = untyped_ema(xs, smooth=0.8, init=0.0)
%timeit _ = typed_ema(xs, smooth=0.8, init=0.0)

ln -s /opt/anaconda/lib/python3.6/site-packages/numpy/core/include/numpy /usr/local/include/numpy
'''


