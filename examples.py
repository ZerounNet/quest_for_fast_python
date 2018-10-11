import json

import numpy as np
import matplotlib.pyplot as plt

def exponential_moving_average(xs, smooth, init):
    smoothed = np.empty_like(xs)
    for t in range(len(xs)):
        if t == 0:
            smoothed[t] = init
        else:
            smoothed[t] = (1-smooth)*xs[t-1] + smooth*smoothed[t-1]
    return smoothed

def json_dumps_loads(data):
    return json.loads(json.dumps(data))

if __name__ == '__main__':
    n_points = 250
    xs = np.linspace(-2*np.pi, 2*np.pi, n_points)
    ys = np.cos(xs) + 0.5*np.random.randn(n_points)
    smoothed02 = exponential_moving_average(ys, smooth=0.2, init=0)
    smoothed08 = exponential_moving_average(ys, smooth=0.8, init=0)

    fig, axis = plt.subplots(figsize=(13, 4))
    axis.scatter(xs, ys, label='Observations', color='black', zorder=-1, alpha=0.5)
    axis.plot(xs, smoothed02, label='Smoothed(0.2)', color='blue')
    axis.plot(xs, smoothed08, label='Smoothed(0.8)', color='red')
    axis.legend()
    fig.savefig('figures/exponential_moving_average')
