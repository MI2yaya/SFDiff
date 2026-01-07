import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math

import sfdiff.configs as diffusion_configs
from sfdiff.model.diffusion.diff import SFDiff
from sfdiff.dataset import get_custom_dataset, get_stored_dataset
from sfdiff.utils import time_splitter, train_test_val_splitter
from dataGeneration import make_kf_matrices_for_sinusoid

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def mse(pred, true): #returns MSE per dimension TBA
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

class SFDiffForecaster:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config["device"]
        self.checkpoint_path = checkpoint_path
        self.model = None

    def _load_model(self, h_fn, R_inv):
        logger.info(f"Loading model checkpoint: {Path(self.checkpoint_path).name}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        scaling = int(self.config["dt"] ** -1)
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

        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        self.model = model
        return(model)
        
    def p_sample_loop_with_snr(self,x_0,num_samples=10):
        start_t = self.model.timesteps
        t = torch.full((num_samples,),start_t-1, dtype=torch.long, device=self.device)
        noise = torch.randn_like(x_0).to(self.device)
        x_0 = x_0.to(self.device)   
        
        x_t = self.model.q_sample(x_0, t, noise=noise)
        
        snr_list = []
        t_list = []
        intermediate_waves = {}

        x_t = x_t.to(self.device)
        x_0 = x_0.to(self.device)
        
        P_signal = torch.mean(x_0 ** 2)
        mid_t = start_t // 2

        for i in reversed(range(start_t)):
            # Compute SNR
            P_noise = torch.mean((x_t - x_0) ** 2)
            snr = 10 * torch.log10(P_signal / (P_noise + 1e-9))
            snr_list.append(snr.item())
            t_list.append(i + 1)

            # Save intermediate waves
            if (i + 1) == start_t or (i + 1) == mid_t:
                intermediate_waves[i + 1] = x_t.detach().cpu().numpy()

            # Sample next step
            t_step = torch.full((1,), i, dtype=torch.long, device=self.device)
            x_t = self.model.p_sample(x=x_t, t=t_step, t_index=i, guidance=False)


        
        x_med = torch.median(x_t, dim=0).values 
        intermediate_waves[0] = x_t 
        P_noise = torch.mean((x_med-x_0) ** 2) 
        snr = 10 * torch.log10(P_signal / (P_noise + 1e-9)) 
        snr_list.append(snr.item()) 
        t_list.append(0)

        return t_list, snr_list, intermediate_waves
    
    
def sincos_to_latlon(sin_lat, cos_lat, sin_lon, cos_lon):
    eps = 1e-6

    sin_lat = torch.as_tensor(sin_lat)
    cos_lat = torch.as_tensor(cos_lat)
    sin_lon = torch.as_tensor(sin_lon)
    cos_lon = torch.as_tensor(cos_lon)

    # Use atan2 directly; no manual renormalization needed
    lat = torch.atan2(sin_lat, cos_lat) * 180.0 / torch.pi
    lon = torch.atan2(sin_lon, cos_lon) * 180.0 / torch.pi

    return lat, lon  
  
    
def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    logger.info(f" {config['dataset']}, {config['data_samples']}, with {config['diffusion_config']}")

    dataset_type,dataset_name = config["dataset"].lower().split(':')
    scaling = int(config['dt']**-1)

    context_length = config["context_length"] * scaling
    prediction_length = config["prediction_length"] * scaling

    num_data_samples=100

    if dataset_type == 'custom':
        base_dataset, generator = get_custom_dataset(dataset_name,
            samples=config['data_samples'],
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            dt=config['dt'],
            q=config['q'],
            r=config['r'],
            observation_dim=config['observation_dim'],
            plot=True
        )
    elif dataset_type == 'dataset':
        base_dataset, generator,config = get_stored_dataset(dataset_name,
            config=config,
            length=config['context_length']+config['prediction_length'],
            plot=True)
    else:
        print(f"Unknown dataset type: {dataset_type}")
        raise NotImplementedError
    
    time_data = time_splitter(base_dataset, context_length, prediction_length)
    split_data = train_test_val_splitter(time_data, num_data_samples, 1/2, 1/2, 0.0)
    train_data = split_data['train']
    test_data = split_data['test']
    
    use_features = config.get('use_features',False)
    

    SFDiff = SFDiffForecaster(config, config['checkpoint_path'])
    SFDiff._load_model(generator.h_fn, generator.R_inv)
    
    skipSFDiffSNR=True
    skipSFDiffCheapCond=True
    skipSFDiffExpensiveCond=False
    plot = False
    skipNAN=True
    
    if skipSFDiffSNR==False:
            trials = 10
            sfdiff_SNRs=[]
            for i in range(trials):
                sample = test_data[i]
                past_state = torch.as_tensor(sample["past_state"], dtype=torch.float32)
                future_state = torch.as_tensor(sample["future_state"], dtype=torch.float32)
                true_state = torch.cat([past_state, future_state], dim=0).unsqueeze(0)
                
                # SFDiff SNRs
                _, snrs, intermediate_waves = SFDiff.p_sample_loop_with_snr(true_state,num_samples=50)
                sfdiff_SNRs.append(snrs[-1]-snrs[0])
                
                if plot and i==0:
                    fig = plt.figure(figsize=(6,8))
                    ax = fig.add_subplot(411)
    
                    ax.plot(snrs, label='SFDiff SNR', color='orange')
                    ax.legend()
                    
                    ax2 = fig.add_subplot(412)
                    ax2.plot(intermediate_waves[list(intermediate_waves.keys())[0]][0,:,0], label='Initial Noisy Sample', color='red')
                    ax3 = fig.add_subplot(413)
                    ax3.plot(intermediate_waves[list(intermediate_waves.keys())[1]][0,:,0], label='Mid Diffusion Sample', color='green')
                    ax4 = fig.add_subplot(414)
                    ax4.plot(true_state[0,:,0].cpu().numpy(), label='True Signal', color='blue')
                    plt.show()
            
            sfdiff_SNRs_avg = np.mean(sfdiff_SNRs, axis=0)
            sfdiff_SNRs_std = np.std(sfdiff_SNRs, axis=0)
            logger.info(f"SFDiff SNR Change over diffusion steps: mean={sfdiff_SNRs_avg:.4f}, std={sfdiff_SNRs_std:.4f}")
            
        
    if skipSFDiffCheapCond == False:
        trials=10
        batch_size=1
        sfdiff_MSEs=[]
        sfdiff_CRPSs=[]
        for i in range(trials):
            series = test_data[i]
            past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)
            
            if use_features:
                past_feat = torch.as_tensor(series["past_features"], dtype=torch.float32)
                future_feat = torch.as_tensor(series["future_features"], dtype=torch.float32)
                features = torch.cat([past_feat, future_feat], dim=0)
            else:
                features = None

            if past_observation.ndim == 2:  # shape (batch, seq_len, dims)
                past_observation = past_observation.unsqueeze(0) 

            if features is not None and features.ndim == 2:
                features = features.unsqueeze(0) 

            y = past_observation.to(device=SFDiff.model.device, dtype=torch.float32)
            features = features.to(device=SFDiff.model.device, dtype=torch.float32) if features is not None else None

            # Generate samples from model
            generated= SFDiff.model.sample_n(
                y=y,
                num_samples=batch_size,
                features=features,
                cheap=False,
                base_strength=.5,
                plot=False,
                guidance=True,
            )
            
            generated_future = np.median(generated[:, -prediction_length:, :].cpu().numpy(),axis=0)
            print(generated_future, future_state.squeeze(0).numpy())
            sfdiff_mse = mse(generated_future, future_state.squeeze(0).numpy())
            print(sfdiff_mse)
            sfdiff_crps = crps(generated[:, -prediction_length:, :].cpu(), future_state)
            sfdiff_MSEs.append(sfdiff_mse)
            sfdiff_CRPSs.append(sfdiff_crps)
            if plot and i==0:
                plt.figure(figsize=(10,5))
                total_length = past_observation.shape[1] + prediction_length
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:past_observation.shape[1]], past_observation.squeeze().cpu().numpy(), label='Past Observations', color='blue')
                plt.plot(time_axis[past_observation.shape[1]:], future_state.squeeze().cpu().numpy(), label='True Future State', color='green')
                plt.plot(time_axis[past_observation.shape[1]:], generated_future, label='SFDiff Prediction', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('SFDiff Cheap Prediction vs True Future State')
                plt.legend()
                plt.show()
            
        sfdiff_mse_avg = np.mean(sfdiff_MSEs)
        sfdiff_mse_std = np.std(sfdiff_MSEs)
        sfdiff_crps_avg = np.mean(sfdiff_CRPSs)
        sfdiff_crps_std = np.std(sfdiff_CRPSs)
        logger.info(f"SFDiff Cheap Guidance + Condition: Future MSE={sfdiff_mse_avg:.4f} ± {sfdiff_mse_std:.4f}, CRPS={sfdiff_crps_avg:.4f} ± {sfdiff_crps_std:.4f}")
        
    if skipSFDiffExpensiveCond == False:
        print("Starting SFDiff Expensive Conditional Sampling")
        trials=20
        batch_size=5
        sfdiff_MSEs=[]
        sfdiff_CRPSs=[]
        for i in range(trials):
            series = test_data[i]
            past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)
            future_state = torch.as_tensor(series["future_state"], dtype=torch.float32)
            if skipNAN and torch.isnan(future_state).any():
                print("Skipping trial with NaN in future state")
                i=i-1
                continue
            if use_features:
                past_feat = torch.as_tensor(series["past_features"], dtype=torch.float32)
                future_feat = torch.as_tensor(series["future_features"], dtype=torch.float32)
                features = torch.cat([past_feat, future_feat], dim=0)
            else:
                features = None

            if past_observation.ndim == 2:  # shape (batch, seq_len, dims)
                past_observation = past_observation.unsqueeze(0) 

            if features is not None and features.ndim == 2:
                features = features.unsqueeze(0) 

            y = past_observation.to(device=SFDiff.model.device, dtype=torch.float32)
            features = features.to(device=SFDiff.model.device, dtype=torch.float32) if features is not None else None

            # Generate samples from model
            generated= SFDiff.model.sample_n(
                y=y,
                num_samples=batch_size,
                features=features,
                cheap=False,
                base_strength=.5,
                plot=False,
                guidance=True,
            )
            generated_future_median = np.median(generated[:, -prediction_length:, :].cpu().numpy(),axis=0)
            generated_future_batch = generated[:, -prediction_length:, :].cpu().numpy()
            if generated.shape[2]>=4: #convert to latlon
                sin_lat = generated_future_median[..., 0]
                cos_lat = generated_future_median[..., 1]
                sin_lon = generated_future_median[..., 2]
                cos_lon = generated_future_median[..., 3]
                lat, lon = sincos_to_latlon(
                    torch.tensor(sin_lat),
                    torch.tensor(cos_lat),
                    torch.tensor(sin_lon),
                    torch.tensor(cos_lon)
                )
                lat = lat.numpy()
                lon = lon.numpy()
                everythingElse = generated_future_median[..., 4:]
                generated_future_median = np.concatenate([lat[..., np.newaxis], lon[..., np.newaxis], everythingElse], axis=-1)

                # For full batch
                sin_lat = generated_future_batch[..., 0]
                cos_lat = generated_future_batch[..., 1]
                sin_lon = generated_future_batch[..., 2]
                cos_lon = generated_future_batch[..., 3]
                lat, lon = sincos_to_latlon(
                    torch.tensor(sin_lat),
                    torch.tensor(cos_lat),
                    torch.tensor(sin_lon),
                    torch.tensor(cos_lon)
                )
                lat = lat.numpy()
                lon = lon.numpy()
                everythingElse = generated_future_batch[..., 4:]
                generated_future_batch = np.concatenate([lat[..., np.newaxis], lon[..., np.newaxis], everythingElse], axis=-1)


                true_sin_lat, true_cos_lat, true_sin_lon, true_cos_lon, true_everythingElse = future_state[:, 0], future_state[:, 1], future_state[:, 2], future_state[:, 3], future_state[:, 4:]
                true_lat, true_lon = sincos_to_latlon(
                    torch.tensor(true_sin_lat),
                    torch.tensor(true_cos_lat),
                    torch.tensor(true_sin_lon),
                    torch.tensor(true_cos_lon)
                )
                future_state = np.concatenate([true_lat.numpy()[:, np.newaxis], true_lon.numpy()[:, np.newaxis], true_everythingElse.numpy()], axis=-1)
                
            # Compute MSE and CRPS
            #print(generated_future_median, future_state)
            sfdiff_mse = mse(generated_future_median, future_state) 
            #print(f"MSE:{sfdiff_mse}")
            sfdiff_crps = crps(generated_future_batch, future_state)
            #print(f"CRPS:{sfdiff_crps}")
            sfdiff_MSEs.append(sfdiff_mse)
            sfdiff_CRPSs.append(sfdiff_crps)
            if plot and i==0:
                plt.figure(figsize=(10,5))
                total_length = past_observation.shape[1] + prediction_length
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:past_observation.shape[1]], past_observation.squeeze().cpu().numpy(), label='Past Observations', color='blue')
                plt.plot(time_axis[past_observation.shape[1]:], future_state, label='True Future State', color='green')
                plt.plot(time_axis[past_observation.shape[1]:], generated_future_median, label='SFDiff Prediction', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('SFDiff Expensive Prediction vs True Future State')
                plt.legend()
                plt.show()
            
        sfdiff_mse_avg = np.nanmean(sfdiff_MSEs)
        sfdiff_mse_median = np.nanmedian(sfdiff_MSEs)
        sfdiff_mse_std = np.nanstd(sfdiff_MSEs)
        sfdiff_crps_avg = np.nanmean(sfdiff_CRPSs)
        sfdiff_crps_median = np.nanmedian(sfdiff_CRPSs)
        sfdiff_crps_std = np.nanstd(sfdiff_CRPSs)
        logger.info(f"SFDiff Expensive Guidance + Condition: \nAverage Future MSE={sfdiff_mse_avg:.4f} ± {sfdiff_mse_std:.4f}, Average CRPS={sfdiff_crps_avg:.4f} ± {sfdiff_crps_std:.4f}\nMedian Future MSE={sfdiff_mse_median:.4f} ± {sfdiff_mse_std:.4f}, Median CRPS={sfdiff_crps_median:.4f} ± {sfdiff_crps_std:.4f}")

    logger.info(f'Done SFDiff')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)