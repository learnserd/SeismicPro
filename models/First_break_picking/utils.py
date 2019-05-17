import numpy as np

def mass_ones(x, thd):
    #arr = np.argmax(x, axis=0)
    arr = softmax(x, axis=0)[1]
    arr = (arr >= thd).astype(np.int)
    arr = np.insert(arr, 0, 0) 
    arr = np.append(arr, 0)
    plus_one = np.argwhere((np.diff(arr)) == 1).flatten()
    minus_one = np.argwhere((np.diff(arr)) == -1).flatten()
    blocks = minus_one - plus_one
    return plus_one[np.argmax(blocks)]

def softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p