import logging
import torch

from pathlib import Path
import numpy as np
import sfdiff.configs as diffusion_configs
from sfdiff.model.diffusion.diff import SFDiff

from pathlib import Path
import torch.nn as nn
import os

BASE_DIR = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SFDiffForecastGenerator:
    """Generates state forecasts from SFDiff models and plots them."""
    
    def __init__(self, config: dict, generator):
        self.config = config
        self.device = config['device']
        self.checkpoint_path = os.path.join(BASE_DIR, config['checkpoint_path'] )
        #self.missing_feat = nn.Parameter(torch.zeros(config.get('num_features',1)))
        self.model = self._load_model(self.checkpoint_path, generator.h_fn, generator.R_inv)
        
    
    def _load_model(self, checkpoint_path: str,h_fn,R_inv):
        logger.info(f"Loading model checkpoint: {Path(checkpoint_path).name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        scaling = int(self.config['dt']**-1)
        context_length = self.config["context_length"] * scaling
        prediction_length = self.config["prediction_length"] * scaling

        model = SFDiff(
            **getattr(diffusion_configs, self.config["diffusion_config"]),
            observation_dim=self.config["observation_dim"],
            context_length=context_length,
            prediction_length=prediction_length,
            lr=self.config["lr"],
            init_skip=self.config["init_skip"],
            h_fn=h_fn,
            R_inv=R_inv,
            modelType=self.config['diffusion_config'].split('_')[1],
            use_transformer=self.config.get('use_transformer',False),
            use_mixer=self.config.get('use_mixer',False),
            use_lags=self.config.get('use_lags',False),
            lag=self.config.get('lag',1),
            num_lags=self.config.get('num_lags',1),
            cross_blocks=self.config.get('cross_blocks',-1),
            use_features=self.config.get('use_features',False),
            num_features=self.config.get('num_features',-1),
            normalize =self.config.get('normalize',False),
            observation_mean=self.config.get('observation_mean',0.0),
            observation_std=self.config.get('observation_std',1.0),
        )

        # Load state_dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in state_dict: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in state_dict: {unexpected}")
        
        model.to(self.device)
        model.eval()
        return model

    def forecast(self, series, num_samples):
        dt = self.config['dt']
        dimension = self.model.observation_dim
        
        past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)
        if self.model.use_features:
            past_feat = torch.as_tensor(series["past_features"], dtype=torch.float32)
            future_feat = torch.as_tensor(series["future_features"], dtype=torch.float32)
            features = torch.cat([past_feat, future_feat], dim=0)
        else:
            features = None

        if past_observation.ndim == 2:  # shape (batch, seq_len, dims)
            past_observation = past_observation.unsqueeze(0) 

        if features is not None and features.ndim == 2:
            features = features.unsqueeze(0) 

        y = past_observation.to(device=self.model.device, dtype=torch.float32)
        features = features.to(device=self.model.device, dtype=torch.float32) if features is not None else None

        # Generate samples from model
        generated= self.model.sample_n(
            y=y,
            num_samples=num_samples,
            features=features,
            cheap=False,
            base_strength=.5,
            plot=False,
            guidance=True,
        )
        
        pred_len = self.model.prediction_length
        generated_batch = generated.cpu().numpy()
        generated_median = np.median(generated.cpu().numpy(),axis=0)  # (T_future, D)
        sin_lat = generated_median[..., 0]
        cos_lat = generated_median[..., 1]
        sin_lon = generated_median[..., 2]
        cos_lon = generated_median[..., 3]
        med_lat, med_lon = self.sincos_to_latlon(
            torch.tensor(sin_lat),
            torch.tensor(cos_lat),
            torch.tensor(sin_lon),
            torch.tensor(cos_lon)
        )
        med_lat = med_lat.numpy()
        med_lon = med_lon.numpy()

        med_extra = generated_median[..., 4:]
        track = []
        for t in range(len(med_lat)):
            track.append({
                "time": float(t * dt),
                "lat": float(med_lat[t]),
                "lon": float(med_lon[t]),
                "windSpeed": (
                    float(med_extra[t, 0]) if med_extra.shape[1] >= 1 and dimension != "2D" else None
                ),
                "pressure": (
                    float(med_extra[t, 1]) if med_extra.shape[1] >= 2 and dimension == "4D" else None
                ),
            })


        batch_sin_lat = generated_batch[..., 0]
        batch_cos_lat = generated_batch[..., 1]
        batch_sin_lon = generated_batch[..., 2]
        batch_cos_lon = generated_batch[..., 3]
        batch_lat, batch_lon = self.sincos_to_latlon(
            torch.tensor(batch_sin_lat),
            torch.tensor(batch_cos_lat),
            torch.tensor(batch_sin_lon),
            torch.tensor(batch_cos_lon)
        )
        batch_lat = batch_lat.numpy()
        batch_lon = batch_lon.numpy()
        everythingElse = generated_batch[..., 4:]
        generated_batch = np.concatenate([batch_lat[..., np.newaxis], batch_lon[..., np.newaxis], everythingElse], axis=-1)

        return {
            "track": track,
            "raw": generated_batch,
        }

    def sincos_to_latlon(self,sin_lat, cos_lat, sin_lon, cos_lon):
        eps = 1e-6

        sin_lat = torch.as_tensor(sin_lat)
        cos_lat = torch.as_tensor(cos_lat)
        sin_lon = torch.as_tensor(sin_lon)
        cos_lon = torch.as_tensor(cos_lon)

        # Use atan2 directly; no manual renormalization needed
        lat = torch.atan2(sin_lat, cos_lat) * 180.0 / torch.pi
        lon = torch.atan2(sin_lon, cos_lon) * 180.0 / torch.pi

        return lat, lon
