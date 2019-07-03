"""Useful functions for tutorials."""
import dill
import numpy as np

def load_speed(file_name, max_time):
    """loading speed from file."""
    with open(file_name, 'r') as file:
        vfunc = file.readline()
        speed = [" ".join(file.readline().replace('\n', '').split()).split(' ') for i in range(2)]
        file.close()
    speed = np.array(speed, dtype=int).ravel()
    speed = speed.reshape(-1, 2)
    time_stamps = speed[:, 0]
    speed_conc = np.concatenate(list(map(lambda a: [a[0]] * a[1], zip(speed[:, 1], 
                                                                      [time_stamps[0],
                                                                       *np.diff(time_stamps)[:-1],
                                                                       max_time - np.max(time_stamps) \
                                                                       +np.diff(time_stamps)[-1]]))))
    return speed_conc

def load_ix(path):
    """Load indices."""
    with open(path, 'rb') as f:
        ix = dill.load(f)
    return ix