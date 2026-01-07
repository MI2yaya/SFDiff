#!/usr/bin/env python3
"""
TSDiff Continual Learning Comparison Plots (State Forecast)
"""

import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection
from pathlib import Path
import argparse
import sfdiff.configs as diffusion_configs
import torch.nn as nn

from sfdiff.model.diffusion.diff import SFDiff
from sfdiff.dataset import get_custom_dataset, get_stored_dataset

from sfdiff.utils import (
    train_test_val_splitter,
    time_splitter,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateForecastPlotter:
    """Generates state forecasts from SFDiff models and plots them."""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = config['device']
        self.checkpoint_path = checkpoint_path
        self.missing_feat = nn.Parameter(torch.zeros(config.get('num_features',1)))
    
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

    @torch.no_grad()
    def generate_state_forecasts(self, dataset_name: str, start_index=0, num_series=1, num_samples=100):
        """Generate forecasts for multiple time series."""
        scaling = int(self.config['dt']**-1)
        testing_samples = num_series-start_index
        
        dataset_type,dataset_name = self.config["dataset"].lower().split(':')
        
        if dataset_type == 'custom':
            dataset, generator = get_custom_dataset(dataset_name,
                samples=testing_samples,
                context_length=self.config["context_length"],
                prediction_length=self.config["prediction_length"],
                dt=self.config['dt'],
                q=self.config['q'],
                r=self.config['r'],
                observation_dim=self.config['observation_dim'],
                plot=False
            )
        elif dataset_type == 'dataset':
            dataset, generator,_ = get_stored_dataset(dataset_name,
                config=self.config,
                length=self.config['context_length']+self.config['prediction_length'],
                plot=False)
        else:
            print(f"Unknown dataset type: {dataset_type}")
            raise NotImplementedError
        
        self.model = self._load_model(self.checkpoint_path,generator.h_fn,generator.R_inv)

        selected_series = dataset[start_index:(start_index + num_series)]
        state_series = time_splitter(selected_series, self.config["context_length"] * scaling, self.config["prediction_length"] * scaling)
        forecasts = []
        for series in state_series:

            past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)

            
            if self.config.get('use_features',False):
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
    
            forecasts.append(generated.cpu().numpy())
            #break # Remove this break to process all series

        return forecasts, state_series

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

    def plot_forecast(self, forecast, series_data, ax, title="Forecast"):
        """Plot a single forecast against ground truth states."""
        past_state = series_data["past_state"]
        future_state = series_data["future_state"]
        total_state = np.concatenate([past_state, future_state], axis=0)
        context_end_idx = int(self.config['context_length'] / self.config['dt'])-1
        
        # Convert forecast sin/cos to lat/lon
        sin_lat = forecast[..., 0]
        cos_lat = forecast[..., 1]
        sin_lon = forecast[..., 2]
        cos_lon = forecast[..., 3]
        

        lat, lon = self.sincos_to_latlon(
            torch.tensor(sin_lat),
            torch.tensor(cos_lat),
            torch.tensor(sin_lon),
            torch.tensor(cos_lon)
        )
        lat = lat.numpy()
        lon = lon.numpy()

        true_sin_lat, true_cos_lat, true_sin_lon, true_cos_lon, true_wind = total_state[:, 0], total_state[:, 1], total_state[:, 2], total_state[:, 3], total_state[:, 4]
        true_lat, true_lon = self.sincos_to_latlon(
            torch.tensor(true_sin_lat),
            torch.tensor(true_cos_lat),
            torch.tensor(true_sin_lon),
            torch.tensor(true_cos_lon)
        )
        
        fig = ax.figure
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes(pos, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        
        # Plot true track
        ax.plot(
            true_lon,
            true_lat,
            color='red',
            linewidth=1.5,
            label='True Track',
            transform=ccrs.PlateCarree()
        )

        # Plot ensemble members
        '''
        for i in range(lat.shape[0]):
            ax.plot(
                lon[i],
                lat[i],
                color="gray",
                alpha=0.2,
                linewidth=1,
                transform=ccrs.PlateCarree()
            )
        '''
        
        # Median track
        lat_med = np.nanmedian(lat, axis=0)
        lon_med = np.nanmedian(lon, axis=0)
        if forecast.shape[-1] >=5:
            wind = forecast[...,4]
            wind_med = np.nanmedian(wind, axis=0)

            # Create line segments for median track colored by wind
            points = np.array([lon_med, lat_med]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap="viridis",
                norm=plt.Normalize(wind_med.min(), wind_med.max()),
                linewidth=2.5,
                transform=ccrs.PlateCarree(),
            )
            lc.set_array(wind_med[:-1])
            ax.add_collection(lc)
        else:
            ax.plot(
                lon_med,
                lat_med,
                color='black',
                linewidth=2.5,
                label='Median Forecast',
                transform=ccrs.PlateCarree()
            )

        ax.plot(
            lon_med[context_end_idx],
            lat_med[context_end_idx],
            marker='o',
            color='blue',
            markersize=8,
            label='End of Context',
            transform=ccrs.PlateCarree()
        )


        # Colorbar
        cbar = plt.colorbar(lc, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Wind speed (kt)")

        ax.set_title(title)
        ax.legend()
        


def create_continual_learning_plots(config, start_series=0, num_series=3):
    """Create plots for multiple time series."""
    checkpoints = {
        "Single Task": config["checkpoint_path"]
    }

    fig, axes = plt.subplots(
        ncols=1, nrows=num_series, figsize=(6, 4 * num_series), squeeze=False, constrained_layout=True
    )
    # Ensure axes is always 2D
    for method_idx, (method_name, ckpt_path) in enumerate(checkpoints.items()):
        plotter = StateForecastPlotter(config, ckpt_path)
        forecasts, series_list = plotter.generate_state_forecasts(
            config["dataset"], start_index=start_series, num_series=num_series, num_samples=100
        )

        for series_idx, forecast in enumerate(forecasts):
            series_data = series_list[series_idx]
            ax = axes[series_idx,0]

            plotter.plot_forecast(
                forecast,  # Use median forecast
                series_data,
                ax,
                title=f"TS{start_series + series_idx}"
            )
            
        logger.info(f"✓ {method_name} plots completed")

    plt.savefig(
        f"continual_learning_states2_{start_series}_to_{start_series+num_series-1}_{config['dataset'].replace(':','_')}.png",
        dpi=300
    )
    plt.close(fig)
    logger.info("Comparison plots saved.")

def main(config_path):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    create_continual_learning_plots(config, start_series=0, num_series=10)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)
