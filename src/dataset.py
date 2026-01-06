import numpy as np
import pandas as pd
from .ppg_filter import bandpass_filter
from .peak_detection import detect_peaks
from .hrv import compute_rr, hrv_features
from .window import sliding_window

def build_dataset(csv_path, fs, label):
    df = pd.read_csv(csv_path)
    ppg = df["PPG"].values

    ppg_filt = bandpass_filter(ppg, fs)
    windows = sliding_window(ppg_filt, fs)

    X, y = [], []

    for w in windows:
        peaks = detect_peaks(w, fs)
        if len(peaks) < 3:
            continue

        rr = compute_rr(peaks, fs)
        feat = hrv_features(rr)

        X.append(list(feat.values()))
        y.append(label)

    return np.array(X), np.array(y)
