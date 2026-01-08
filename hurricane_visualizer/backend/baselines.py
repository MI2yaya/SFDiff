import numpy as np

def run_baseline_model(model_name, obs, dt, dimension,horizon=5):

    obs = np.asarray(obs)  # (T_past, D)
    T,D = obs.shape

    if D < 2:
        raise ValueError("Observation dimension must be at least 2 (lat, lon).")

    # --- LAT/LON ONLY (degrees) ---
    lat_past = obs[:, 0]
    lon_past = obs[:, 1]

    # Estimate velocity (last 2 steps)
    dlat = lat_past[-1] - lat_past[-2]
    dlon = lon_past[-1] - lon_past[-2]

    pred = np.zeros((T+horizon,D),dtype=np.float32)
    pred[:T]=obs
    track=[]

    for t in range(T):
        entry = {
            "time": float(t * dt),
            "lat": float(obs[t, 0]),
            "lon": float(obs[t, 1]),
            "windSpeed": float(obs[t, 2]) if D >= 3 and dimension != "2D" else None,
            "pressure": float(obs[t, 3]) if D >= 4 and dimension == "4D" else None,
        }
        if D > 2:
            entry["features"] = obs[t, 2:].tolist()
        track.append(entry)

    for k in range(horizon):
        t = T + k

        # Linear extrapolation
        lat = lat_past[-1] + (k + 1) * dlat
        lon = lon_past[-1] + (k + 1) * dlon

        pred[t, 0] = lat
        pred[t, 1] = lon

        # Persist all remaining dimensions
        if D > 2:
            pred[t, 2:] = obs[-1, 2:]

        track_entry = {
            "time": float((k + 1) * dt),
            "lat": float(lat),
            "lon": float(lon),
            "windSpeed": float(pred[t, 2]) if D >= 3 and dimension != "2D" else None,
            "pressure": float(pred[t, 3]) if D >= 4 and dimension == "4D" else None,
        }

        # Optional semantic labels if present
        if D > 2:
            track_entry["features"] = pred[t, 2:].tolist()

        track.append(track_entry)

    return {
        "track": track,      # future only
        "raw": pred,         # future only
    }
