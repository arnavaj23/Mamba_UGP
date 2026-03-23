# ============================================================
# IMPORTS + PATH FIX
# ============================================================
import sys
import os

# 🔥 load your custom mamba_ssm + pscan from dataset
sys.path.insert(
    0,
    "/kaggle/input/datasets/theroyalseal/10kmph-dataset-and-mamba-architecture"
)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import yaml

from dataset.csi_dataset import CSIDataset
from models import MambaResidualDiffusion
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# TRAIN FUNCTION
# ============================================================
def train():

    # -----------------------------
    # CONFIG (edit if needed)
    # -----------------------------
    config_path = "Mamba_UGP/DDPM-Pytorch/config/default.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # -----------------------------
    # SCHEDULER
    # -----------------------------
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    # -----------------------------
    # DATASET
    # -----------------------------
    dataset = CSIDataset(
        dataset_config['data_path'],
        window=dataset_config['window']
    )

    loader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    model = MambaResidualDiffusion(
        in_dim=model_config['in_dim'],
        hidden=model_config['hidden_dim'],
        layers=model_config['num_layers']
    ).to(device)

    model.train()

    # -----------------------------
    # OUTPUT DIR
    # -----------------------------
    os.makedirs(train_config['task_name'], exist_ok=True)

    ckpt_path = os.path.join(
        train_config['task_name'],
        train_config['ckpt_name']
    )

    # -----------------------------
    # LOAD CHECKPOINT (optional)
    # -----------------------------
    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = Adam(model.parameters(), lr=train_config['lr'])

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    for epoch in range(train_config['num_epochs']):

        losses = []

        for window, target in tqdm(loader):

            optimizer.zero_grad()

            window = window.to(device)   # [B,T,D]
            target = target.to(device)   # [B,D]

            # -------------------------
            # last state
            # -------------------------
            last_state = window[:, -1, :]

            # -------------------------
            # clean signal
            # -------------------------
            x0 = target

            # -------------------------
            # noise
            # -------------------------
            noise = torch.randn_like(x0, device=device) * 0.3

            # -------------------------
            # timestep
            # -------------------------
            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                (x0.shape[0],),
                device=device
            ).long()

            # -------------------------
            # forward diffusion
            # -------------------------
            noisy_x = scheduler.add_noise(x0, noise, t)

            # -------------------------
            # inject into sequence
            # -------------------------
            x_seq = window.clone()
            x_seq[:, -1, :] = noisy_x

            # -------------------------
            # predict noise
            # -------------------------
            noise_pred = model(x_seq, t, last_state)

            # -------------------------
            # loss
            # -------------------------
            loss = ((noise_pred - noise) ** 2).mean()

            losses.append(loss.item())

            # -------------------------
            # backward
            # -------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.6f}")

        torch.save(model.state_dict(), ckpt_path)

    print("Training complete!")


# ============================================================
# RUN
# ============================================================
train()