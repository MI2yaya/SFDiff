# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from sfdiff.arch.backbones import SequenceBackbone, UNetBackbone
from sfdiff.model.diffusion._base import SFDiffBase
from sfdiff.utils import make_diffusion_gif,make_lagged


class SFDiff(SFDiffBase):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        observation_dim,
        h_fn,
        R_inv,
        init_skip=True,
        lr=1e-3,
        dropout_rate=0.01,
        modelType='s4',
        use_transformer=False,
        use_mixer=False,
        use_lags=False,
        lag=1,
        num_lags=1,
        cross_blocks=-1,
        use_features=False,
        num_features=-1,
        normalize=False,
        observation_mean=0.0,
        observation_std=1.0,
    ):
        super().__init__(
            timesteps=timesteps,
            diffusion_scheduler=diffusion_scheduler,
            context_length=context_length,
            prediction_length=prediction_length,
            lr=lr,
            dropout_rate=dropout_rate,
            use_lags=use_lags,
            lag=lag,
            num_lags=num_lags,
            use_features=use_features,
            num_features=num_features,
            normalize=normalize,
            observation_mean=torch.tensor(observation_mean,dtype=torch.float32),
            observation_std=torch.tensor(observation_std,dtype=torch.float32),
        )
        if use_lags:
            lagged_dim = observation_dim * num_lags
            print("Using lagged data with lags:", lag, "num_lags:", num_lags)
        else:
            lagged_dim = observation_dim
            
        self.use_lags = use_lags
        self.lag = lag
        self.num_lags = num_lags
        
        self.use_features=use_features
        self.num_features=num_features
        
        backbone_parameters["observation_dim"] = lagged_dim
        backbone_parameters['output_dim'] = lagged_dim
        print(backbone_parameters)

        self.modelType=modelType
        
        if modelType=='s4' or modelType=="s5":
            print(f'Running {modelType}')
            backbone_parameters["dropout"] = dropout_rate
            self.backbone = SequenceBackbone(
                **backbone_parameters,
                init_skip=init_skip,
                use_transformer=use_transformer,
                use_mixer=use_mixer,
                cross_blocks=cross_blocks if cross_blocks > 0 else observation_dim,
                use_features=use_features,
                num_features=num_features,
                modelType=modelType,
            )
            
        elif modelType=='unet':
            print('Running unet')
            self.backbone = UNetBackbone(
                **backbone_parameters
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_parameters['type']}")
        
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]
        self.observation_dim = observation_dim
        self.h_fn = h_fn
        self.R_inv = R_inv


    @torch.no_grad()
    def sample_n(
        self,
        y,
        num_samples: int = 1,
        cheap=True,
        base_strength=0.1,
        plot=False,
        guidance=True,
        features=None,
        
    ):
        device = next(self.backbone.parameters()).device
        obs_dim = self.observation_dim
        lagged_dim = obs_dim * self.num_lags if self.use_lags else obs_dim

        pad_len = self.lag * (self.num_lags - 1) if self.use_lags else 0
        total_len = pad_len + self.context_length + self.prediction_length

    

        # Prepare y and masks
        if y is not None:
            if self.normalize:
                print("Normalizing with,",self.observation_mean,self.observation_std)
                y = (y - self.observation_mean.view(1, 1, -1)) / self.observation_std.view(1, 1, -1)
            else:
                y = y

            nan_mask = ~torch.isnan(y)
            y_clean = torch.nan_to_num(y, nan=0.0)
            y_clean = y_clean.repeat(num_samples, 1, 1)
            nan_mask = nan_mask.repeat(num_samples, 1, 1)
            known_len = y_clean.shape[1]
        else:
            y_clean = None
            nan_mask = None
            known_len = 0

        if features is not None:
            features = features.to(device)
            missing = torch.isnan(features)
            features = torch.nan_to_num(features, nan=0.0)
            features = features + missing * self.missing_feat

        # Initialize noise in lagged space
        samples = torch.randn((num_samples, total_len, lagged_dim), device=device)

        # Known values
        if y is not None:
            mask = torch.zeros_like(samples)
            known_full = torch.zeros_like(samples)
            mask[:, pad_len:pad_len+known_len, :obs_dim] = 1.0
            known_full[:, pad_len:pad_len+known_len, :obs_dim] = y_clean
        else:
            mask = None
            known_full = None

        # Reverse diffusion
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x_prev_uncond = self.p_sample(
                x=samples,
                t=t,
                t_index=i,
                y=y_clean,      # pass lagged y for guidance
                y_mask=nan_mask,
                h_fn=self.h_fn,
                R_inv=self.R_inv,
                base_strength=base_strength,
                cheap=cheap,
                plot=plot,
                guidance=guidance,
                features=features
            )

            if mask is not None and i > 0:
                t_prev = torch.full((num_samples,), i - 1, dtype=torch.long, device=device)
                known_prev = self.q_sample(known_full, t_prev)
                samples = (mask * known_prev) + ((1 - mask) * x_prev_uncond)
            else:
                samples = x_prev_uncond

        # Reconstruct full-length forecast (original_seq_len) with padding for initial missing timesteps due to lagging
        samples_full = samples[:, pad_len:, :obs_dim]

        if self.normalize:
            samples_full = samples_full * self.observation_std.view(1, 1, -1) + self.observation_mean.view(1, 1, -1)
        
        if plot:
            make_diffusion_gif()

        return samples_full

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for rate, state_dict in zip(self.ema_rate, self.ema_state_dicts):
            update_ema(state_dict, self.backbone.state_dict(), rate=rate)


def update_ema(target_state_dict, source_state_dict, rate=0.99):
    with torch.no_grad():
        for key, value in source_state_dict.items():
            ema_value = target_state_dict[key]
            ema_value.copy_(
                rate * ema_value + (1.0 - rate) * value.cpu(),
                non_blocking=True,
            )
