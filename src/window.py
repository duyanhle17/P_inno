import numpy as np

def sliding_window(signal, fs, window_sec=30, overlap=0.5):
    win_len = int(window_sec * fs)
    step = int(win_len * (1 - overlap))

    windows = []
    for start in range(0, len(signal) - win_len, step):
        windows.append(signal[start:start + win_len])

    return windows
