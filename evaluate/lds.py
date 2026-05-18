import torch

class LatentDensityMetric:
    def __init__(self, sigma=5.0):
        self.sigma = sigma
    
    def __call__(self, gen_repr, train_repr):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        Z_g = torch.tensor(gen_repr, dtype=torch.float32, device=device)
        Z_train = torch.tensor(train_repr, dtype=torch.float32, device=device)
        
        dist_sq = torch.cdist(Z_g, Z_train, p=2).pow(2)  # [N_gen, N_train]
        density_matrix = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        density_scores = density_matrix.mean(dim=1)  # [N_gen]
        return density_scores.mean().item()