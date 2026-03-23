import torch
import torch.nn as nn
from mamba_ssm import Mamba, MambaConfig

class MambaResidualDiffusion(nn.Module):

    def __init__(self, in_dim, hidden, layers):
        super().__init__()

        self.freq_embed = nn.Parameter(torch.zeros(1,1,in_dim))

        self.input_proj = nn.Linear(in_dim, hidden)

        self.freq_mixer = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )

        self.temporal_conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)

        self.mamba = Mamba(MambaConfig(
            d_model = hidden,
            n_layers = layers
        ))

        self.fast_block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim)
        )

        self.slow_block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

        # 🔥 NEW: timestep embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        nn.init.normal_(self.freq_embed,std=0.02)


    def forward(self, x_t, t, last_state):

        # x_t: [B,T,D]
        # last_state: [B,D]

        B,T,D = x_t.shape

        # -----------------------------
        # 1. Centering (IMPORTANT)
        # -----------------------------
        last_state = last_state.unsqueeze(1)  # [B,1,D]
        x = x_t - last_state

        # -----------------------------
        # 2. Add frequency embedding
        # -----------------------------
        x = x + self.freq_embed

        # -----------------------------
        # 3. Input projection
        # -----------------------------
        x = self.input_proj(x)

        # -----------------------------
        # 4. Timestep embedding (NEW)
        # -----------------------------
        t = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.t_embed(t).unsqueeze(1)

        x = x + t_emb

        # -----------------------------
        # 5. Mixer
        # -----------------------------
        x = x + self.freq_mixer(x)

        # -----------------------------
        # 6. Temporal conv
        # -----------------------------
        xt = x.transpose(1,2)

        x1 = self.temporal_conv1(xt)
        x2 = self.temporal_conv2(xt)

        x = (x1 + x2).transpose(1,2)

        # -----------------------------
        # 7. Mamba forward + backward
        # -----------------------------
        xf = self.mamba(x)
        xb = self.mamba(x.flip(1)).flip(1)

        x = xf + xb

        # -----------------------------
        # 8. Mean pooling
        # -----------------------------
        ctx = torch.mean(x, dim=1)

        # -----------------------------
        # 9. Fast / Slow heads
        # -----------------------------
        fast = self.fast_block(ctx)
        slow = self.slow_block(ctx)

        # -----------------------------
        # 🔥 FINAL OUTPUT = NOISE ε
        # -----------------------------
        eps = self.alpha * fast + self.beta * slow

        return eps   # [B,D]