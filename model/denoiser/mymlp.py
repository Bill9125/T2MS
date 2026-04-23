import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"
    def forward(self, t):
        t = t * 100.0
        t = t.unsqueeze(-1)
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)
        return embedding

class MLPlayer(nn.Module):
    def __init__(self, dim=64, cond_dim=128, seq_len=6):
        super().__init__()
        # 改用 elementwise_affine=False 因為我們交給 AdaLN 來接管縮放和平移
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(seq_len, elementwise_affine=False)
        
        # AdaLN block: 負責預測 norm1 和 norm2 的 scale 與 shift
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * dim)
        )
        self.adaLN2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * seq_len)
        )
        
        # DiT 論文中最關鍵的一步：將 AdaLN 最後一層初始化為 0，讓整個殘差塊訓練初期等於 Identity
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)
        nn.init.constant_(self.adaLN2[-1].weight, 0)
        nn.init.constant_(self.adaLN2[-1].bias, 0)
        
        self.self_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.SiLU(),
            nn.Linear(256, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.SiLU(),
            nn.Linear(256, seq_len)
        )

    def forward(self, x, cond, pos_emb):
        # x is [B, 64, 6]. Permute for sequence processing: [B, 6, 64]
        x = x.permute(0, 2, 1)
        
        # 1. Self Attention 區塊
        shift_1, scale_1, shift_2, scale_2 = self.adaLN(cond).chunk(4, dim=-1)
        shift_1, scale_1 = shift_1.unsqueeze(1), scale_1.unsqueeze(1)
        shift_2, scale_2 = shift_2.unsqueeze(1), scale_2.unsqueeze(1)
        
        normed_x1 = self.norm1(x) * (1 + scale_1) + shift_1
        attn_in = normed_x1 + pos_emb  # 注入位置編碼
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in)
        x = x + attn_out
        
        # 2. 特徵 MLP 區塊
        normed_x2 = self.norm2(x) * (1 + scale_2) + shift_2
        x = x + self.mlp(normed_x2)
        
        # 3. 序列混合 MLP2 區塊
        x = x.permute(0, 2, 1)  # 回到原視角 [B, 64, 6]
        
        shift_3, scale_3 = self.adaLN2(cond).chunk(2, dim=-1)
        shift_3, scale_3 = shift_3.unsqueeze(1), scale_3.unsqueeze(1)  # [B, 1, 6]
        
        normed_x3 = self.norm3(x) * (1 + scale_3) + shift_3
        x = x + self.mlp2(normed_x3)
        
        return x

class myMLP(nn.Module):
    def __init__(self, in_channels=64, cond_dim=128, seq_len=6):
        super().__init__()
        self.time_emb = TimeEmbedding(dim=cond_dim)
        
        # Global 的位置編碼
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, in_channels))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        self.layers = nn.ModuleList([MLPlayer(dim=in_channels, cond_dim=cond_dim, seq_len=seq_len) for _ in range(8)])
        
    def forward(self, input, t, text_input=None):
        # 統整出全局的聯合條件向量 (Cond)
        t_emb = self.time_emb(t)
        cond = t_emb
        if text_input is not None:
            # text_input is (B, 128), t_emb is (B, 128)
            cond = cond + text_input
            
        for layer in self.layers:
            input = layer(input, cond, self.pos_emb)
            
        return input
