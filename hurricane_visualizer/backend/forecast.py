import numpy as np
import torch
from datetime import datetime
from typing import Dict

from SFDiff import SFDiffForecastGenerator
from metrics import compute_metrics
from baselines import run_baseline_model

def make_timeseries(seq: np.ndarray, dt: float, dimension: str, start_time=0.0):
    T,D = seq.shape
    track=[]
    for t in range(T):
        track_entry = {
            "time": float(start_time + t * dt),
            "lat": float(seq[t,0]),
            "lon": float(seq[t,1]),
            "windSpeed": float(seq[t,2]) if D >=3 and dimension != "2D" else None,
            "pressure": float(seq[t,3]) if D >=4 and dimension == "4D" else None,
        }
        track.append(track_entry)
    return track

def generate(
    sim_config: Dict,
    dataset_id: str,
    model_config: Dict,
    test_data: list,
    generator,
) -> Dict:
    """
    Main backend entrypoint.
    Safe for FastAPI / Flask / serverless usage.
    """

    variant = sim_config["sfdiffVariant"]
    dimension = sim_config["sfdiffDimension"]
    active_models = sim_config["activeModels"]

    SFDiffModel = SFDiffForecastGenerator(
        config=model_config, generator=generator
    )

    track_data: Dict[str, list] = {}
    metrics_data: Dict[str, dict] = {}
    
    future_state = test_data[0]["future_state"]
    
    true_sin_lat, true_cos_lat, true_sin_lon, true_cos_lon, true_everythingElse = future_state[:, 0], future_state[:, 1], future_state[:, 2], future_state[:, 3], future_state[:, 4:]
    true_lat, true_lon = SFDiffModel.sincos_to_latlon(
        torch.tensor(true_sin_lat),
        torch.tensor(true_cos_lat),
        torch.tensor(true_sin_lon),
        torch.tensor(true_cos_lon)
    )
    future_state = np.concatenate([true_lat.numpy()[:, np.newaxis], true_lon.numpy()[:, np.newaxis], true_everythingElse.numpy()], axis=-1)

    prediction_length = model_config.get("prediction_length",5)
    
    if "sfdiff" in active_models or True:
        forecast = SFDiffModel.forecast(
            series=test_data[0],
            num_samples=10,
        )

        track_data["sfdiff"] = forecast["track"]
        metrics_data["sfdiff"] = compute_metrics(
            forecast["raw"][:, -prediction_length:, :],
            future_state,
            calc_median=True
        )
        
    print("SFDiff forecast complete.")   
    
    obs = test_data[0]['past_observation']
    
    obs_sin_lat, obs_cos_lat, obs_sin_lon, obs_cos_lon, obs_everythingElse = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4:]
    obs_lat, obs_lon = SFDiffModel.sincos_to_latlon(
        torch.tensor(obs_sin_lat),
        torch.tensor(obs_cos_lat),
        torch.tensor(obs_sin_lon),
        torch.tensor(obs_cos_lon)
    )
    obs = np.concatenate([obs_lat.numpy()[:, np.newaxis], obs_lon.numpy()[:, np.newaxis], obs_everythingElse.numpy()], axis=-1)
    
    for model_name in active_models:
        print(f"Running baseline model: {model_name}")
        if model_name == "sfdiff":
            continue


        baseline_forecast = run_baseline_model(
            model_name=model_name,
            obs=obs,
            horizon=5,
            dt=model_config["dt"],
            dimension=dimension,
        )
        raw = baseline_forecast["raw"]

        if raw.ndim == 2:
            raw = raw[None, ...]
        elif raw.ndim != 3:
            raise ValueError(f"Unexpected baseline raw shape: {raw.shape}")

        baseline_forecast["raw"] = raw

        track_data[model_name] = baseline_forecast["track"]
        metrics_data[model_name] = compute_metrics(
            baseline_forecast["raw"][:, -prediction_length:, :],
            future_state,
            calc_median=True,
            )

    track_data["past"] = make_timeseries(obs, model_config["dt"], dimension,start_time=0.0)
    last_obs = obs[-1:, :]  # shape (1, D)

    # Stack last_obs on top of future_state
    truth_seq = np.vstack([last_obs, future_state])  # shape (H+1, D)

    track_data["groundTruth"] = make_timeseries(truth_seq, model_config["dt"], dimension,start_time=obs.shape[0]*model_config.get("dt",1))

    return {
        "id": f"{dataset_id}_{variant}_{dimension}",
        "name": dataset_id,
        "description": f"SFDiff-{variant} {dimension} Forecast",
        "track": track_data,
        "metrics": metrics_data,
        "generatedAt": datetime.utcnow().isoformat(),
    }