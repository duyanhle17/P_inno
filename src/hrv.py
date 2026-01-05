import numpy as np

def compute_rr(peaks, fs):
    return np.diff(peaks) / fs

def hrv_features(rr):
    mean_rr = np.mean(rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    hr = 60 / mean_rr

    return {
        "HR": hr,
        "MeanRR": mean_rr,
        "SDNN": sdnn,
        "RMSSD": rmssd
    }

