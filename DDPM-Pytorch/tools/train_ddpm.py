import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.csi_dataset import CSIDataset
from models import MambaResidualDiffusion
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

    # -----------------------------
    # Load config
    # -----------------------------
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # -----------------------------
    # Scheduler
    # -----------------------------
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    # -----------------------------
    # Dataset
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
        pin_memory=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = MambaResidualDiffusion(
        in_dim=model_config['in_dim'],
        hidden=model_config['hidden_dim'],
        layers=model_config['num_layers']
    ).to(device)

    model.train()

    # -----------------------------
    # Output folder
    # -----------------------------
    os.makedirs(train_config['task_name'], exist_ok=True)

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    ckpt_path = os.path.join(
        train_config['task_name'],
        train_config['ckpt_name']
    )

    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(train_config['num_epochs']):

        losses = []

        for window, target in tqdm(loader):

            optimizer.zero_grad()

            window = window.to(device)   # [B,T,D]
            target = target.to(device)   # [B,D]

            # -------------------------
            # Last state
            # -------------------------
            last_state = window[:, -1, :]   # [B,D]

            # -------------------------
            # x0 (already amp/sin/cos)
            # -------------------------
            x0 = target

            # -------------------------
            # Sample noise
            # -------------------------
            noise = torch.randn_like(x0) * 0.5

            # -------------------------
            # Sample timestep
            # -------------------------
            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                (x0.shape[0],),
                device=device
            )

            # -------------------------
            # Add noise
            # -------------------------
            noisy_x = scheduler.add_noise(x0, noise, t)

            # -------------------------
            # Build sequence input
            # -------------------------
            x_seq = window.clone()
            x_seq[:, -1, :] = noisy_x

            # -------------------------
            # Predict noise
            # -------------------------
            noise_pred = model(x_seq, t, last_state)

            # -------------------------
            # Loss
            # -------------------------
            loss = criterion(noise_pred, noise)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.6f}")

        torch.save(model.state_dict(), ckpt_path)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()

    train(args)