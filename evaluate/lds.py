import torch
import numpy as np

class LatentDensityMetric:
    def __init__(self, sigma=5.0):
        self.sigma = sigma
    
    def __call__(self, gen_repr, train_repr):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # L2 Normalization to bound distances in [0, 2] and prevent underflow in high-dimensional space
        gen_repr = gen_repr / (np.linalg.norm(gen_repr, axis=1, keepdims=True) + 1e-8)
        train_repr = train_repr / (np.linalg.norm(train_repr, axis=1, keepdims=True) + 1e-8)
        
        Z_g = torch.tensor(gen_repr, dtype=torch.float32, device=device)
        Z_train = torch.tensor(train_repr, dtype=torch.float32, device=device)
        
        dist_sq = torch.cdist(Z_g, Z_train, p=2).pow(2)  # [N_gen, N_train]
        density_matrix = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        density_scores = density_matrix.mean(dim=1)  # [N_gen]
        return density_scores.mean().item()