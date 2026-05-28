import numpy as np
from scipy.linalg import sqrtm

class CFIDMetric:
    def __init__(self):
        self.name = "C-FID"

    def __call__(self, y_real, y_gen, x_cond):
        """
        Calculate Conditional FID based on the formula:
        CFID = ||m_y - m_y_hat||^2 + Tr((Cyx - Cy_hat_x)Cxx^-1(Cxy - Cx_y_hat)) + Tr(Cyy|x + Cy_hat_y_hat|x - 2(Cyy|x^1/2 * Cy_hat_y_hat|x * Cyy|x^1/2)^1/2)
        """
        # L2 Normalization to bound distances in [0, 2] and prevent underflow in high-dimensional space
        y_real = y_real / (np.linalg.norm(y_real, axis=1, keepdims=True) + 1e-8)
        y_gen = y_gen / (np.linalg.norm(y_gen, axis=1, keepdims=True) + 1e-8)
        x_cond = x_cond / (np.linalg.norm(x_cond, axis=1, keepdims=True) + 1e-8)

        # 1. Means
        m_y = y_real.mean(axis=0)
        m_y_hat = y_gen.mean(axis=0)
        
        # 2. Covariances
        def get_stats(y, x):
            yx = np.concatenate([y, x], axis=1)
            C = np.cov(yx, rowvar=False)
            dy = y.shape[1]
            C_yy = C[:dy, :dy]
            C_xx = C[dy:, dy:]
            C_yx = C[:dy, dy:]
            C_xy = C_yx.T
            return C_yy, C_xx, C_yx, C_xy

        C_yy, C_xx, C_yx, C_xy = get_stats(y_real, x_cond)
        C_yhat_yhat, _, C_yhat_x, C_x_yhat = get_stats(y_gen, x_cond)

        # Pre-calculate C_xx_inv (using pseudo-inverse for stability)
        C_xx_inv = np.linalg.pinv(C_xx)

        # --- Part 1: Global Mean Difference ---
        diff_mean = np.sum((m_y - m_y_hat)**2)

        # --- Part 2: Contextual Correlation Difference ---
        diff_corr = (C_yx - C_yhat_x)
        part2 = np.trace(diff_corr @ C_xx_inv @ diff_corr.T)

        # --- Part 3: Conditional Overlap ---
        # Conditional Covariances
        C_yy_cond = C_yy - C_yx @ C_xx_inv @ C_xy
        C_yhat_yhat_cond = C_yhat_yhat - C_yhat_x @ C_xx_inv @ C_x_yhat
        
        # Fréchet Distance part for the conditional covariances
        covmean = sqrtm(C_yy_cond.dot(C_yhat_yhat_cond))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        part3 = np.trace(C_yy_cond + C_yhat_yhat_cond - 2.0 * covmean)

        return diff_mean + part2 + part3
