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

def mse(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)

    mask = ~np.isnan(pred) & ~np.isnan(true)
    if mask.sum() == 0:
        return np.nan  # or raise, depending on preference

    return np.mean((pred[mask] - true[mask]) ** 2)

def crps(pred, true):
    pred = torch.as_tensor(pred, dtype=torch.float32)
    true = torch.as_tensor(true, dtype=torch.float32)
    

    # CASE: both 1-D and same length -> treat as sequence batch
    if pred.ndim == 1 and true.ndim == 1 and pred.shape[0] == true.shape[0]:
        # turn into [B=T, N=1, F=1] and [B=T, 1, 1]
        pred = pred.view(-1, 1, 1)
        true = true.view(-1, 1, 1)
    else:
        # Generic promotion to [B, N, F] and [B, 1, F]
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        elif pred.ndim == 2:
            pred = pred.unsqueeze(-1)               # [B, N, 1]
        # true handling
        if true.ndim == 0:
            true = true.unsqueeze(0).unsqueeze(-1)  # scalar -> [1,1,1]
        elif true.ndim == 1:
            true = true.unsqueeze(1).unsqueeze(-1)  # [B,1,1]  (if B matches pred batch)
        elif true.ndim == 2:
            true = true.unsqueeze(1)               # [B,1,F]

    B, T, D = pred.shape

    # Mask NaNs in truth (shared across ensemble)
    valid_mask = ~torch.isnan(true[0])    # [T, D]

    crps_vals = []

    for d in range(D):
        mask = valid_mask[:, d]
        if mask.sum() == 0:
            continue

        pred_d = pred[:, mask, d]          # [B, T_valid]
        true_d = true[:, mask, d]          # [B, T_valid]

        # All ensemble members share the same truth
        true_d = true_d[0]                 # [T_valid]

        # E|X - y|
        term1 = torch.mean(
            torch.abs(pred_d - true_d.unsqueeze(0)),
            dim=0
        )                                  # [T_valid]

        # E|X - X'|
        diffs = torch.abs(
            pred_d.unsqueeze(0) - pred_d.unsqueeze(1)
        )                                  # [B, B, T_valid]
        term2 = 0.5 * torch.mean(diffs, dim=(0, 1))

        crps_d = term1 - term2
        crps_d = torch.clamp(crps_d, min=0.0)

        crps_vals.append(crps_d.mean())

    if len(crps_vals) == 0:
        return float("nan")

    return torch.mean(torch.stack(crps_vals)).item()

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
    

    SFDiff = SFDiffForecaster(config, config['checkpoint_path'])
    SFDiff._load_model(generator.h_fn, generator.R_inv)
    
    skipSFDiffSNR=True
    skipSFDiffCheapCond=True
    skipSFDiffExpensiveCond=False
    plot = True
    
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
            
            if "past_features" in series and series["past_features"] is not None \
            and "future_features" in series and series["future_features"] is not None:
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
        trials=10
        batch_size=1
        sfdiff_MSEs=[]
        sfdiff_CRPSs=[]
        for i in range(trials):
            series = test_data[i]
            past_observation = torch.as_tensor(series["past_observation"], dtype=torch.float32)
            future_state = torch.as_tensor(series["future_state"], dtype=torch.float32)
            
            if "past_features" in series and series["past_features"] is not None \
            and "future_features" in series and series["future_features"] is not None:
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
            # Compute MSE and CRPS
            print(generated_future, future_state.squeeze(0).numpy())
            sfdiff_mse = mse(generated_future, future_state.squeeze(0).numpy())
            print(sfdiff_mse)
            sfdiff_crps = crps(generated_future, future_state.squeeze(0).numpy())
            print(sfdiff_crps)
            sfdiff_MSEs.append(sfdiff_mse)
            sfdiff_CRPSs.append(sfdiff_crps)
            if plot: #and i==0:
                plt.figure(figsize=(10,5))
                total_length = past_observation.shape[1] + prediction_length
                time_axis = np.arange(total_length) * config['dt']
                plt.plot(time_axis[:past_observation.shape[1]], past_observation.squeeze().cpu().numpy(), label='Past Observations', color='blue')
                plt.plot(time_axis[past_observation.shape[1]:], future_state.squeeze().cpu().numpy(), label='True Future State', color='green')
                plt.plot(time_axis[past_observation.shape[1]:], generated_future, label='SFDiff Prediction', color='red')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title('SFDiff Expensive Prediction vs True Future State')
                plt.legend()
                plt.show()
            
        sfdiff_mse_avg = np.mean(sfdiff_MSEs)
        sfdiff_mse_std = np.std(sfdiff_MSEs)
        sfdiff_crps_avg = np.mean(sfdiff_CRPSs)
        sfdiff_crps_std = np.std(sfdiff_CRPSs)
        logger.info(f"SFDiff Expensive Guidance + Condition: Future MSE={sfdiff_mse_avg:.4f} ± {sfdiff_mse_std:.4f}, CRPS={sfdiff_crps_avg:.4f} ± {sfdiff_crps_std:.4f}")
    
    logger.info(f'Done SFDiff')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    args = parser.parse_args()

    main(args.config)