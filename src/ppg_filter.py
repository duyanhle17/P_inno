import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, low=0.5, high=5):
    """
    Lọc PPG:
    - 0.5–5 Hz ~ 30–300 BPM
    """
    nyq = 0.5 * fs
    b, a = butter(2, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, signal)


