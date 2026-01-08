import numpy as np
import torch

def mse(pred, true): 
    pred = torch.as_tensor(pred, dtype=torch.float32)
    true = torch.as_tensor(true, dtype=torch.float32)
    assert pred.shape == true.shape, "Shapes of pred and true must match"
    assert pred.ndim == 2, "pred and true must be 2D arrays (time, dimensions)"
    pred = np.asarray(pred)
    true = np.asarray(true)

    mask = ~np.isnan(pred) & ~np.isnan(true)
    
    sq_err = (true - pred) ** 2
    sq_err[~mask] = np.nan

    return np.mean(sq_err, axis=0)

def crps(pred, true):
    #pred (B,T,D), true (T,D)
    pred = torch.as_tensor(pred, dtype=torch.float32)
    true = torch.as_tensor(true, dtype=torch.float32)

    if pred.ndim != 3:
        raise ValueError("pred must have shape (B, T, D)")
    if true.ndim != 2:
        raise ValueError("true must have shape (T, D)")
    if pred.shape[1:] != true.shape:
        raise ValueError("pred shape (B,T,D) must match true shape (T,D)")

    B, T, D = pred.shape
    crps = np.full((D,), float("nan"))

    for d in range(D):
        mask = ~torch.isnan(true[:, d])          # (T,)
        if not mask.any():
            continue

        # (B, T_valid)
        pred_d = pred[:, mask, d]
        true_d = true[mask, d]                    # (T_valid,)

        # E|X - y|
        term1 = torch.mean(
            torch.abs(pred_d - true_d.unsqueeze(0)),
            dim=0                                 # average over ensemble
        )                                         # (T_valid,)

        # E|X - X'|
        diffs = torch.abs(
            pred_d.unsqueeze(0) - pred_d.unsqueeze(1)
        )                                         # (B, B, T_valid)
        term2 = 0.5 * torch.mean(diffs, dim=(0, 1))  # (T_valid,)

        crps[d] = torch.mean(term1 - term2).numpy()
    return crps

def compute_metrics(pred: np.ndarray, truth: np.ndarray,calc_median=False) -> dict:
    '''
    Compute all metrics for given prediction and truth arrays.
    '''
    
    #print(pred,truth)
    crps_vals = crps(pred, truth)
    if calc_median:
        pred = np.median(pred, axis=0)
    mse_vals = mse(pred, truth)
    
    latlon = slice(0, 2)

    return {
        "mse": float(np.nanmean(mse_vals[latlon])),
        "crps": float(np.nanmean(crps_vals[latlon])),
    }