import numpy as np

def exponential_moving_average(xs, smooth, init):
    smoothed = []
    for t in range(len(xs)):
        if t == 0:
            smoothed.append(init)
        else:
            smoothed.append(
                (1-smooth)*xs[t-1] + smooth*smoothed[-1]
            )
    return smoothed
xs = np.random.randn(10000).tolist()

'''
%timeit _ = exponential_moving_average(xs, smooth=0.8, init=0.0)
'''
