import numpy as np
import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    def __init__(self, path, window=30):

        raw = np.load(path)   # [U, T, 2, 32, 32]

        U, T, _, H, W = raw.shape

        # -----------------------------
        # convert real + imag → complex
        # -----------------------------
        real = raw[:, :, 0, :, :]
        imag = raw[:, :, 1, :, :]

        complex_data = real + 1j * imag   # [U, T, 32, 32]

        # -----------------------------
        # flatten spatial dims
        # -----------------------------
        complex_data = complex_data.reshape(U, T, -1)   # [U, T, D]

        # -----------------------------
        # convert → amp + sin + cos
        # -----------------------------
        amp = np.abs(complex_data)
        phase = np.angle(complex_data)

        sin = np.sin(phase)
        cos = np.cos(phase)

        data = np.concatenate([amp, sin, cos], axis=-1)   # [U, T, 3D]

        # -----------------------------
        # normalize
        # -----------------------------
        data = data / (np.std(data) + 1e-6)

        self.data = data.astype(np.float32)
        self.window = window

        self.U = U
        self.T = T

    def __len__(self):
        return self.U * (self.T - self.window - 1)

    def __getitem__(self, idx):

        # -----------------------------
        # map idx → user + time
        # -----------------------------
        u = idx // (self.T - self.window - 1)
        t = idx % (self.T - self.window - 1)

        # -----------------------------
        # window + target
        # -----------------------------
        window = self.data[u, t : t + self.window]     # [T, D]
        target = self.data[u, t + self.window]         # [D]

        return torch.from_numpy(window), torch.from_numpy(target)