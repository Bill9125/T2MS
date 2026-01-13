import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import numpy as np
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_sinusoidal_positional_embeddings(num_positions, d_model):
    position = torch.arange(num_positions).unsqueeze(1)  # shape: (num_positions, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(
        0)  # shape: (1, d_model/2)

    pos_embedding = torch.zeros(num_positions, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    return pos_embedding.unsqueeze(0)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"
    def forward(self, t):
        # t = t * 100.0
        t = t.unsqueeze(-1)

        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)

        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)
        return embedding

################################################
#               Embedding Layers               #
################################################

class LatentEmbedding(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        self.dim = embed_dim
        self.embedding2d = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(6, 6),
            stride=(6, 6),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        B, _, M, N = x.shape
        x = self.embedding2d(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class InverseLatentEmbedding(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        self.dim = embed_dim
        self.inv_embedding2d = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=(6, 6),
            stride=(6, 6),
        )
        self.fc1 = nn.Linear(60, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        B, K, C = x.shape
        x = x.transpose(1, 2).reshape(B, self.dim, 1, K)
        x = self.inv_embedding2d(x)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


#################################################################################
#                                 Core Model                                #
#################################################################################

class Transformerlayer(nn.Module):
    def __init__(self, ):
        super().__init__()
        d_model = 128 #64
        mlp_ratio = 2.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        # self.fc1 = nn.Linear(64, 128)
        # self.fc2 = nn.Linear(128, 64)


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        # x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.fc2(x)


        return x


class Transformer(nn.Module):
    def __init__(self, dim, embedding_dim=64):
        super().__init__()
        # dim: Time length (T)
        # embedding_dim: Channel dimensions (C)
        self.seq_len = dim
        self.input_channels = embedding_dim
        
        # Patching Config
        # 我們將 2D Patching 改為 1D Time-Patching
        # 為什麼？因為特徵維度 (Channels) 不應該被切分，它們在同一時間點代表身體的整體狀態。
        # 我們只在時間軸上進行切分。
        self.patch_size = 2  # 每次處理 2 個時間步
        self.embed_dim = 128 # Transformer 內部維度

        # 計算 Patch 數量 = 時間長度 / Patch 大小
        self.num_patches = self.seq_len // self.patch_size
        
        # 每個 Patch 的原始攤平維度 = Channel * Patch_Size
        self.patch_input_dim = self.input_channels * self.patch_size

        # Patch Embedding: 將每個 Patch 投影到 Transformer 內部維度
        # 輸入: [B, Patch_Input_Dim, Num_Patches] -> 輸出: [B, Embed_Dim, Num_Patches]
        # 這裡使用 Conv1d 作為 Patch Projection，步長 = Patch_Size 實現不重疊切分
        self.patch_proj = nn.Conv1d(
            in_channels=self.input_channels, 
            out_channels=self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        pos_embed = get_sinusoidal_positional_embeddings(self.num_patches, self.embed_dim)
        self.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)
        
        self.time_emb = TimeEmbedding(dim=self.embed_dim)

        self.layers = nn.ModuleList([Transformerlayer() for _ in range(6)])
        
        self.ln = nn.LayerNorm(self.embed_dim)
        
        # Output Projection: 將 Transformer 輸出還原回原始維度
        # Linear: Embed_Dim -> Patch_Input_Dim
        self.output_proj = nn.Linear(self.embed_dim, self.patch_input_dim)

        self.initialize_weights()

    def forward(self, input: torch.Tensor, t: torch.Tensor, text_input):
        """
        input: (B, Channels, Time) -> 注意這裡預期輸入是 [B, C, T]
        t: (B,) tensor of diffusion timesteps
        text_input: Text embedding
        """
        # input shape: [B, C, T]
        # 1. Patchify using Conv1d
        # x shape: [B, Embed_Dim, Num_Patches]
        x = self.patch_proj(input) 
        
        # Change to sequence format: [B, Num_Patches, Embed_Dim]
        x = x.transpose(1, 2)
        
        # 2. Add Positional Embedding
        x = x + self.pos_embed

        # 3. Process Time & Text Conditioning
        t_emb = self.time_emb(t)
        c = t_emb
        if text_input is not None:
            c = c + text_input
            
        # 4. Transformer Blocks
        for layer in self.layers:
            x = layer(x, c)

        # 5. Final output processing
        x = self.ln(x) # [B, Num_Patches, Embed_Dim]
        
        # 6. Unpatchify (还原)
        # 投影回原始 Patch 大小: [B, Num_Patches, C * Patch_Size]
        x = self.output_proj(x)
        
        # Reshape 回 [B, Num_Patches, C, Patch_Size]
        B, N, _ = x.shape
        x = x.view(B, N, self.input_channels, self.patch_size)
        
        # Permute to [B, C, Num_Patches, Patch_Size]
        x = x.permute(0, 2, 1, 3)
        
        # Flatten time dim: [B, C, T]
        x = x.reshape(B, self.input_channels, self.patch_size * self.num_patches)
        
        return x

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
