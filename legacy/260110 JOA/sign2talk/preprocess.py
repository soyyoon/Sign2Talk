import numpy as np

def resample_to_fixed_length(seq_feats: np.ndarray, target_len: int) -> np.ndarray:
    T, F = seq_feats.shape
    if T == target_len:
        return seq_feats.astype(np.float32)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(x_new, x_old, seq_feats[:, f]).astype(np.float32)
    return out

def predict_on_fixed30(model, fixed30_feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.expand_dims(fixed30_feats, axis=0)
    x = (x - mean) / std
    return model.predict(x, verbose=0)[0]
