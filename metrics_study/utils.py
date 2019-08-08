import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import patches


def get_windowed_spectrogram_dists(smgr, smgl, dist_fn='sum_abs', time_frame_width=100, noverlap=None, window=('tukey', 0.25)):
    
    kwargs = dict(window=window, nperseg=time_frame_width, noverlap=noverlap)
    f, t, Sxx_l = signal.spectrogram(smgl, **kwargs)
    f, t, Sxx_r = signal.spectrogram(smgr, **kwargs)

    if callable(dist_fn):  # res(sl, sr)
        res = dist_fn(Sxx_l, Sxx_r)
    elif dist_fn == 'max_abs':
        res = np.abs(Sxx_l - Sxx_r).max(axis=1)
    elif dist_fn == 'sum_abs':
        res = np.sum(np.abs(Sxx_l - Sxx_r), axis=1)
    elif dist_fn == 'sum_sq':
        res = np.sum(np.abs(Sxx_l - Sxx_r) ** 2, axis=1)
#         elif mode == 'corr'
    else:
        raise NotImplemented('modes other than max_abs, sum_abs, sum_sq not implemented yet')

    return res
       
def draw_modifications_dist(modifications, traces_frac=0.1, distances='sum_abs', 
                            vmin=None, vmax=None, figsize=(15, 15), 
                            time_frame_width=100, noverlap=None, window=('tukey', 0.25), n_cols=None):
    
    nx, ny = 1, len(modifications) 
    if n_cols is not None:
        nx, ny = int(np.ceil(ny / n_cols)), n_cols 
    
    fig, axs = plt.subplots(nx, ny, figsize=figsize)  
    axs = axs.flatten()
    
    origin, _ = modifications[0]
    n_traces, n_ts = origin.shape
    n_use_traces = int(n_traces*traces_frac)
    
    if isinstance(distances, str) or callable(distances):
        distances = (distances, )
    
    for i, (mod, description) in enumerate(modifications):
        distances_strings = []
        for dist_fn in distances:
            dist_m = get_windowed_spectrogram_dists(mod[0:n_use_traces], origin[0:n_use_traces], dist_fn=dist_fn, 
                                                   time_frame_width=time_frame_width, noverlap=noverlap, window=window)
            dist = np.mean(dist_m)
            distances_strings.append("{}: {:.4}".format(dist_fn, dist))

        axs[i].imshow(mod.T, vmin=vmin, vmax=vmax, cmap='gray')
        rect = patches.Rectangle((0, 0), n_use_traces, n_ts, edgecolor='r', facecolor='none', lw=1)
        axs[i].add_patch(rect)
        axs[i].set_title("{},\ndistances from original are:\n{}".format(description, '\n'.join(distances_strings)))
        
def get_cv(arrs, q=0.95):
    return np.abs(np.quantile(np.stack(item for item in arrs), q))
