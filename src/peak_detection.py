import numpy as np
from scipy.signal import find_peaks

def detect_peaks(ppg, fs):
    peaks, _ = find_peaks(
        ppg,
        distance=fs*0.4,   # ≥150 BPM
        prominence=0.5
    )
    return peaks
