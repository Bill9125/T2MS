from ast import arg
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .core import BaseModel

class PositionalEncoding(nn.Module):
    """
    Positional encoding layer that adds unique encoding to each position in the sequence
    
    Parameter explanations:
    - d_model (int): Model dimension, feature dimension for all embeddings
    - max_len (int): Maximum sequence length, determines encoding table size
    - dropout (float): Dropout ratio to prevent overfitting
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x (torch.Tensor): Input features shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Features with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AdaLN(nn.Module):
    """
    Adaptive layer normalization that dynamically adjusts normalization parameters based on conditions
    
    Parameter explanations:
    - d_model (int): Model dimension
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.ln = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2)
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization
        
        Args:
            x (torch.Tensor): Input features shape (batch_size, seq_len, d_model)
            condition (torch.Tensor): Condition vector shape (batch_size, d_model)
        
        Returns:
            torch.Tensor: Normalized features
        """
        x_ln = self.ln(x)
        gamma_beta = self.mlp(condition)
        
        gamma = gamma_beta[:, :gamma_beta.size(1)//2]
        beta = gamma_beta[:, gamma_beta.size(1)//2:]
        
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return gamma * x_ln + beta

class TimeSeriesEncoder(nn.Module):
    """
    Transformer encoder for encoding input time series
    
    Parameter explanations:
    - n_features (int): Feature dimension of input sequence (number of variables)
    - d_model (int): Model dimension [Recommended: 256-512]
    - num_layers (int): Number of Transformer encoder layers [Recommended: 4-8]
    - d_ff (int): Hidden layer dimension of feed-forward network [Recommended: d_model*4]
    - num_heads (int): Number of attention heads [Recommended: 8]
    - dropout (float): Dropout ratio [Recommended: 0.1]
    - max_seq_len (int): Maximum sequence length [Recommended: 2000]
    """
    
    def __init__(
        self,
        n_features: int,
        flow_dim: int = 128,
        num_layers: int = 6,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2000
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.flow_dim = flow_dim
        
        # Multi-feature embedding layer: project n_features dimension to d_model dimension
        # Example: 32 input features -> embed to 512 dimensions
        self.value_embedding = nn.Linear(n_features, flow_dim)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.zeros_(self.value_embedding.bias)
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.embedding_ln = nn.LayerNorm(flow_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=flow_dim,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=flow_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(
        self,
        time_series: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode time series
        
        Args:
            time_series (torch.Tensor): Input time series
                shape: (batch_size, seq_len, n_features)
            
            src_key_padding_mask (Optional[torch.Tensor]): Padding mask
                shape: (batch_size, seq_len)
                dtype: bool
        
        Returns:
            encoder_output (torch.Tensor): Encoded representation
                shape: (batch_size, seq_len, d_model)
        """
        
        # Project multi-feature vectors to d_model dimension
        embedded = self.value_embedding(time_series)
        embedded = self.embedding_dropout(embedded)
        embedded = self.embedding_ln(embedded)
        
        # Add positional encoding
        encoded_pos = self.positional_encoding(embedded)
        
        # Pass through Transformer encoder layers
        x = self.transformer_encoder(
            encoded_pos,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return x
    
class AdaptiveLinear(nn.Module):
    def __init__(self, out_features, max_in_features=2048):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, max_in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        in_dim = x.size(-1)
        return F.linear(x, self.weight[:, :in_dim], self.bias)

class ConditionFusionModule(nn.Module):
    """
    Module to fuse text conditions and length information
    
    Parameter explanations:
    - flow_dim (int): Flow dimension
    """
    
    def __init__(
        self,
        flow_dim: int = 128,
        max_text_length: int = 512,
        text_embedding_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.flow_dim = flow_dim
        
        # Text condition projection layer
        max_text_features = max_text_length * text_embedding_dim  # ✅ 正确计算
        self.text_projection = AdaptiveLinear(
            out_features=flow_dim,
            max_in_features=max_text_features,  # ✅ 用正确的值
        )
        self.condition_projection = nn.Linear(flow_dim, flow_dim)
        nn.init.xavier_uniform_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(flow_dim * 2, flow_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(flow_dim * 4, flow_dim)
        )
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.fusion_ln = nn.LayerNorm(flow_dim)
        
        # Adaptive layer normalization
        self.ada_ln = AdaLN(flow_dim)
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        
        # Project text condition
        batch_size, _, _ = encoder_output.shape
        text_emb_flat = text_embedding.reshape(batch_size, -1)
        text_cond = self.text_projection(text_emb_flat)
        
        # Expand to sequence dimension
        _, seq_len, _ = encoder_output.shape
        
        text_cond_expanded = text_cond.unsqueeze(1).expand(-1, seq_len, -1)
        # Concatenate all information
        combined = torch.cat([
            encoder_output,
            text_cond_expanded,
        ], dim=-1)
        
        # Fusion layer
        fused = self.fusion(combined)
        fused = fused + encoder_output
        fused = self.fusion_ln(fused)
        condition = self.condition_projection(text_cond)
        # Adaptive layer normalization
        conditioned = self.ada_ln(fused, condition)
        
        return conditioned


# ============================================================================
# 5. Time Series Decoder (Transformer)
# ============================================================================

class TimeSeriesDecoder(nn.Module):
    """
    Transformer decoder for autoregressive time series generation
    
    Parameter explanations:
    - n_features (int): Feature dimension of output sequence (number of variables)
    - d_model (int): Model dimension [Recommended: 256-512]
    - num_layers (int): Number of Transformer decoder layers [Recommended: 4-8]
    - d_ff (int): Hidden layer dimension of feed-forward network [Recommended: d_model*4]
    - num_heads (int): Number of attention heads [Recommended: 8]
    - dropout (float): Dropout ratio [Recommended: 0.1]
    - max_seq_len (int): Maximum sequence length [Recommended: 2000]
    """
    
    def __init__(
        self,
        n_features: int,
        num_layers: int = 6,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2000,
        flow_dim: int = 128,
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.flow_dim = flow_dim
        self.n_features = n_features
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=flow_dim,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=flow_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection layer: project to n_features dimension
        self.output_projection = nn.Linear(flow_dim, n_features)
        self.input_projection = nn.Linear(n_features, flow_dim)
        nn.init.xavier_uniform_(self.input_projection.weight)
        
    def _generate_causal_mask(
        self,
        size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate causal mask (lower triangular matrix)
        
        Args:
            size (int): Sequence length
            device (torch.device): Device where tensor resides
        
        Returns:
            torch.Tensor: Causal mask matrix shape (size, size)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, memory, target_seq, tgt_key_padding_mask=None):
        """
        訓練模式：使用 Teacher Forcing
        memory: Encoder/Fusion 的輸出 (B, seq_len, flow_dim)
        target_seq: Ground Truth 目標序列 (B, seq_len, n_features)
        """
        batch_size, seq_len, _ = target_seq.shape
        
        # 1. 準備 Decoder 輸入： [Start_Token, x_0, x_1, ..., x_{N-1}]
        # 需要把 target_seq 投影到 flow_dim (如果你沒有對應的 embedding 層，這裡需要一個 Linear)
        # 假設我們需要一個 Linear 層把 n_features 轉回 flow_dim 供輸入使用
        
        # 將數值特徵投影到隱藏層維度
        target_emb = self.input_projection(target_seq) # (B, seq_len, flow_dim)
        bos = torch.zeros(batch_size, 1, self.flow_dim, device=target_seq.device)
        decoder_input = torch.cat([bos, target_emb[:, :-1, :]], dim=1)
        
        # 2. 加位置編碼
        tgt = self.positional_encoding(decoder_input)
        
        # 3. 產生 Causal Mask (防止偷看未來)
        tgt_mask = self._generate_causal_mask(seq_len, device=target_seq.device)
        
        # 4. Transformer Decoder (一次性平行計算)
        output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 5. 投影回輸出特徵
        prediction = self.output_projection(output)
        return prediction

    def generate(self, memory):
        """
        推理模式：Autoregressive 生成 (原本你的 forward 邏輯移到這裡)
        """
        batch_size = memory.size(0)
        seq_len = memory.size(1) # 假設生成長度跟 memory 一樣
        
        decoder_input = torch.zeros(batch_size, 1, self.flow_dim, device=memory.device)
        generated_tokens = []
        
        for step in range(seq_len):
            pos_encoded = self.positional_encoding(decoder_input)
            causal_mask = self._generate_causal_mask(decoder_input.size(1), memory.device)
            
            # 這裡其實很慢，因為每次都重算，但為了邏輯正確先這樣寫
            out = self.transformer_decoder(pos_encoded, memory, tgt_mask=causal_mask)
            last_out = out[:, -1:, :] # 取最後一個
            
            # 預測數值
            pred_val = self.output_projection(last_out)
            generated_tokens.append(pred_val)
            
            # 下一步輸入 (需要投影回 flow_dim)
            # 注意：這裡需要跟 forward 一樣的 input_projection
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(self.n_features, self.flow_dim).to(memory.device)
            
            next_input = self.input_projection(pred_val)
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
            
        return torch.cat(generated_tokens, dim=1)


# ============================================================================
# 6. Complete Model
# ============================================================================

class AttentionSeq2SeqAutoencoder(BaseModel):
    """
    Attention-Based Seq2Seq Autoencoder complete model
    
    Supports generation of multi-feature time series
    
    Parameter explanations:
    - n_features (int): Feature dimension of input/output sequence (number of variables)
    - flow_dim (int): Flow dimension, controls width of all layers [Recommended: 512]
    - num_encoder_layers (int): Number of encoder layers [Recommended: 6]
    - num_decoder_layers (int): Number of decoder layers [Recommended: 6]
    - num_heads (int): Number of attention heads [Recommended: 8]
    - d_ff (int): Hidden layer dimension of feed-forward network [Recommended: 2048 (d_model*4)]
    - max_length (int): Maximum sequence length [Recommended: 2000]
    - dropout (float): Dropout ratio [Recommended: 0.1]
    """
    
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.n_features = args.input_dim
        self.flow_dim = args.flow_dim
        
        # Time series encoder
        self.encoder = TimeSeriesEncoder(
            n_features=args.input_dim,
            flow_dim=args.flow_dim,
            num_layers=args.num_encoder_layers,
            d_ff=args.d_ff,
            num_heads=args.num_heads,
        )
        
        # Condition fusion module
        self.condition_fusion = ConditionFusionModule(
            flow_dim=args.flow_dim,
        )
        
        # Time series decoder
        self.decoder = TimeSeriesDecoder(
            n_features=args.input_dim,
            flow_dim=args.flow_dim,
            num_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            num_heads=args.num_heads,
        )
    
    def forward(
        self,
        time_series: torch.Tensor,
        text_embedding: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            time_series (torch.Tensor): Input time series
                shape: (batch_size, input_seq_len, n_features)
            
            text_embedding (torch.Tensor): Text condition embedding
                shape: (batch_size, text_encoder_dim)
            
            src_key_padding_mask (Optional[torch.Tensor]): Input padding mask
                shape: (batch_size, input_seq_len)
            
            tgt_key_padding_mask (Optional[torch.Tensor]): Output padding mask
                shape: (batch_size, target_seq_len)
        
        Returns:
            generated_sequence (torch.Tensor): Generated time series
                shape: (batch_size, target_len, n_features)
        """
        
        # Step 1: Encode input time series
        x = self.encoder(
            time_series,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Step 2: Fuse text condition and target length
        x = self.condition_fusion(
            x,
            text_embedding
        )
        # Step 3: Decode and generate target sequence
        x = self.decoder(
            x,
            time_series,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return x
    
    def forward_inference(
        self,
        time_series: torch.Tensor,
        text_embedding: torch.Tensor,
        target_length: Optional[int] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference mode with autoregressive generation
        
        Args:
            time_series: (batch_size, seq_len, n_features)
            target_length: Target sequence length (if None, use input length)
            src_key_padding_mask: Optional padding mask
        
        Returns:
            generated_sequence: (batch_size, target_length, n_features)
        """
        
        # Step 1: Encode input time series
        x = self.encoder(
            time_series,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Step 2: Fuse text condition
        # x = self.condition_fusion(x, text_embedding)
        
        # Step 3: Autoregressive generation
        x = self.decoder.generate(memory=x)
        
        return x
    
    def shared_eval(self, batch, text_embedding, optimizer, mode):
        if mode == 'train':
            optimizer.zero_grad()
            
            # 1. Encoder
            z = self.encoder(batch)
            
            # 2. Fusion
            # z = self.condition_fusion(z, text_embedding)
            
            # 3. Decoder (Training Mode - Teacher Forcing)
            # 傳入 batch 作為目標序列，讓 Decoder 學習 "給定 z 和 前t個真實值，預測第t+1個值"
            recon = self.decoder(memory=z, target_seq=batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            
        elif mode == 'val' or mode == 'test':
            self.eval()
            with torch.no_grad():
                recon = self.forward_inference(batch, text_embedding)
                loss = F.mse_loss(recon, batch)
                
        return loss, recon
