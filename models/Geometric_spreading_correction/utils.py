"""File with helpful functions for shpere difference correction."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, hilbert

def draw_gain(sample, window, xbounds=None, ybounds=None):
    """Draw difference of amplitude by time."""
    if isinstance(sample, (tuple, list)):
        sample = np.array(sample)
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    h_sample = []
    for trace in sample:
        hilb = hilbert(trace).real
        env = (trace**2 + hilb**2)**.5
        h_sample.append(env)
    h_sample = np.array(h_sample)
    mean_sample = np.mean(h_sample, axis=0)
    max_val = np.max(mean_sample)
    dt_val = (-1) * (max_val / mean_sample)
    result = medfilt(dt_val, window)

    if xbounds is None:
        xbounds = (min(result)-min(result)*1.1, max(result)+min(result)*1.1)
    elif not isinstance(xbounds, (list, tuple, np.ndarray)):
        raise ValueError('xbounds should be list/tuple or numpy array with lenght 2'\
                         +', not {}'.format(type(xbounds)))
    elif len(xbounds) != 2:
        raise ValueError('xbounds should has lenght 2 not {}'.format(len(xbounds)))

    if ybounds is None:
        ybounds = (len(result)+100, -100)
    elif not isinstance(ybounds, (list, tuple, np.ndarray)):
        raise ValueError('ybounds should be list/tuple or numpy array with lenght 2'\
                         +', not {}'.format(type(ybounds)))
    elif len(ybounds) != 2:
        raise ValueError('ybounds should has lenght 2 not {}'.format(len(ybounds)))

    plt.figure(figsize=(10, 8))
    plt.plot(result, range(len(result)))
    plt.title('Amplitude gain')
    plt.xlim(xbounds)
    plt.ylim(ybounds)
    plt.xlabel('Maxamp/Amp')
    plt.ylabel('Time')
    plt.show()
