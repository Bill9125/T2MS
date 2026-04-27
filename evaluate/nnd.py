import numpy as np
from scipy.spatial.distance import cdist

class NNDMetric:
    def __init__(self):
        self.name = "NND"

    def __call__(self, eval_repr, train_repr):
        """
        Calculate Nearest Neighbor Distance (NND) - Novelty Score
        eval_repr: [N_eval, Dim]
        train_repr: [N_train, Dim]
        """
        # L2 Normalization
        eval_repr = eval_repr / (np.linalg.norm(eval_repr, axis=1, keepdims=True) + 1e-8)
        train_repr = train_repr / (np.linalg.norm(train_repr, axis=1, keepdims=True) + 1e-8)

        # Euclidean distance matrix [N_eval, N_train]
        distances = cdist(eval_repr, train_repr, metric='euclidean')
        
        # Find minimum distance for each evaluation sample
        min_distances = np.min(distances, axis=1)
        return np.mean(min_distances)
