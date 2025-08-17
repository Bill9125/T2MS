import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pretrained.core import BaseModel

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels, num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(num_residual_hiddens, num_hiddens, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super().__init__()
        # 下採樣 x4: 100 -> 50 -> 25
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv1d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        """
        inputs: [B, 52, 100] 或 [B, 100, 52]，自動轉為 [B, 52, 100]
        """
        x = inputs
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, L] or [B, L, C], got {x.shape}")

        # 若是 [B, 100, 52] 則轉成 [B, 52, 100]
        if x.shape[1] != 52 and x.shape[2] == 52:
            x = x.permute(0, 2, 1).contiguous()

        # 這裡開始是 [B, 52, 100]
        x = self._conv_1(x)     # -> [B, H/2, 50]
        x = F.relu(x)
        x = self._conv_2(x)     # -> [B, H, 25]
        x = F.relu(x)
        x = self._conv_3(x)     # -> [B, H, 25]
        x = self._residual_stack(x)  # -> [B, H, 25]
        x = self._pre_vq_conv(x)     # -> [B, E, 25]
        before = x                      # 為了 cross_loss（25）
        z = F.interpolate(x, size=30, mode='linear', align_corners=True)  # -> [B, E, 30]
        return z, before

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=52):
        super().__init__()
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._conv_trans_1 = nn.ConvTranspose1d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose1d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs, length):
        """
        inputs: [B, E, 30]
        length: 原始序列長度（例如 100）
        回到 encoder 的 25，再經兩次上採樣 -> length
        """
        x = F.interpolate(inputs, size=int(length / 4), mode='linear', align_corners=True)  # -> [B, E, 25]
        after = x  # 與 encoder 的 before（[B, E, 25]）對齊
        x = self._conv_1(x)         # -> [B, H, 25]
        x = self._residual_stack(x) # -> [B, H, 25]
        x = self._conv_trans_1(x)   # -> [B, H/2, 50]
        x = F.relu(x)
        x = self._conv_trans_2(x)   # -> [B, 52, 100]
        return x, after

class vqvae(BaseModel):
    def __init__(self, args):
        super().__init__()
        num_hiddens = args.block_hidden_size
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.res_hidden_size
        embedding_dim = args.embedding_dim

        # in_channels=52（多特徵）
        self.encoder = Encoder(
            in_channels=52,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            embedding_dim=embedding_dim
        )
        # decoder 輸出 52 通道
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            out_channels=52
        )

    def _ensure_channel_first(self, x):
        # 接受 [B, 52, 100] 或 [B, 100, 52]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, L] or [B, L, C], got {x.shape}")
        if x.shape[1] != 52 and x.shape[2] == 52:
            x = x.permute(0, 2, 1).contiguous()
        return x

    def shared_eval(self, batch, optimizer, mode):
        batch = self._ensure_channel_first(batch)  # -> [B, 52, 100]
        L = batch.shape[-1]

        if mode == 'train':
            optimizer.zero_grad()
            z, before = self.encoder(batch)                    # z: [B, E, 30], before: [B, E, 25]
            data_recon, after = self.decoder(z, length=L)      # recon: [B, 52, 100], after: [B, E, 25]
            recon_error = F.mse_loss(data_recon, batch)
            cross_loss = F.mse_loss(before, after)
            loss = recon_error + cross_loss
            loss.backward()
            optimizer.step()
        else:  # 'val' or 'test'
            with torch.no_grad():
                z, before = self.encoder(batch)
                data_recon, after = self.decoder(z, length=L)
                recon_error = F.mse_loss(data_recon, batch)
                cross_loss = F.mse_loss(before, after)
                loss = recon_error + cross_loss

        return loss, recon_error, data_recon, z

    def forward(self, x):
        x = self._ensure_channel_first(x)   # [B, 52, 100]
        L = x.shape[-1]
        z, _ = self.encoder(x)              # [B, E, 30]
        out, _ = self.decoder(z, length=L)  # [B, 52, 100]
        return out
