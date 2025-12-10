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
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, flow_dim):
        super().__init__()
        self.flow_dim= flow_dim
        # 下採樣 x4: N -> 50 -> 25
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
        x = inputs # -> [B, 10, T]

        x = self._conv_1(x)     # -> [B, H/2, T/2]
        x = F.relu(x)
        x = self._conv_2(x)     # -> [B, H, T/4]
        x = F.relu(x)
        x = self._conv_3(x)     # -> [B, H, T/4]
        x = self._residual_stack(x)
        x = self._pre_vq_conv(x)     # -> [B, H/2, T/2]
        before = x
        z = F.interpolate(x, size=self.flow_dim, mode='linear', align_corners=True)  # -> [B, H/2, 300]
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
        x = F.interpolate(inputs, size=int(length / 4), mode='linear', align_corners=True)  # -> [B, E, L/4]
        after = x
        x = self._conv_1(x)         # -> [B, H, T/2]
        x = self._residual_stack(x) # -> [B, H, T/2]
        x = self._conv_trans_1(x)   # -> [B, H/2, T]
        
        x = F.relu(x)
        x = self._conv_trans_2(x)   # -> [B, 10, L+e]
        x = F.interpolate(x, size=length, mode='linear', align_corners=True)
        return x, after

class vqvae(BaseModel):
    def __init__(self, args):
        super().__init__()
        num_hiddens = args.block_hidden_size
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.res_hidden_size
        embedding_dim = args.embedding_dim
        flow_dim = args.flow_dim
        input_dim = args.input_dim

        # in_channels=10（多特徵）
        self.encoder = Encoder(
            in_channels=input_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            embedding_dim=embedding_dim,
            flow_dim=flow_dim
        )
        # decoder 輸出 10 通道
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            out_channels=input_dim
        )
    
    def norm(self, x_nfT):
        # 正規化每個特徵序列到 [0, 1]
        for i in range(x_nfT.size(0)):  # 對每個特徵
            feat = x_nfT[i]  # [T]
            min_val = feat.min()
            max_val = feat.max()
            if max_val > min_val:  # 避免除以零
                x_nfT[i] = (feat - min_val) / (max_val - min_val)
            else:
                x_nfT[i] = 0.0  # 如果序列值全部相同，設為 0
        return x_nfT

    def shared_eval(self, batch, optimizer, mode): # pyright: ignore[reportIncompatibleMethodOverride]
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
        L = x.shape[-1]
        z, _ = self.encoder(x)              # [B, E, 30]
        out, _ = self.decoder(z, length=L)  # [B, 52, 100]
        return out
