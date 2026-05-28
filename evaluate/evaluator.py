import numpy as np
import torch
from evaluate.utils import show_with_start_divider, show_with_end_divider
from evaluate.cfid import CFIDMetric
from evaluate.nnd import NNDMetric
from evaluate.lds import LatentDensityMetric

def evaluate_data(args, ori_data, gen_data, index, result, vae_encoder=None, train_repr_my=None, cond_embs=None):
    show_with_start_divider(f"Evaluation with settings: {args}")

    method_list = args.method_list
    device = args.device

    if not isinstance(method_list, list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]

    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None
    if ori_data.shape != gen_data.shape:
        print(f'Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}.')
        show_with_end_divider('Error: Data shape mismatch.')
        return None, None, None
    
    with torch.no_grad():
        ori_tensor = torch.tensor(ori_data).float().to(device)
        gen_tensor = torch.tensor(gen_data).float().to(device)
        
        # Get VAE features [Batch, Channel, Time]
        ori_features, _ = vae_encoder(ori_tensor)
        gen_features, _ = vae_encoder(gen_tensor)
        
        # For C-FID: Average over time [Batch, Channel]
        ori_repr_fid = ori_features.mean(dim=-1).cpu().numpy()
        gen_repr_fid = gen_features.mean(dim=-1).cpu().numpy()
        
        # For Novelty (NND): Flatten [Batch, Channel * Time]
        ori_repr_nnd = ori_features.flatten(start_dim=1).cpu().numpy()
        gen_repr_nnd = gen_features.flatten(start_dim=1).cpu().numpy()

    result[index] = {}
    n_folds = getattr(args, 'n_folds', 1)
    
    if n_folds <= 1:
        if 'C-FID' in method_list:
            cfid_metric_fn = CFIDMetric()
            result[index]['C-FID'] = cfid_metric_fn(ori_repr_fid, gen_repr_fid, cond_embs)

        if 'NND' in method_list:
            nnd_metric_fn = NNDMetric()
            result[index]['Novelty-Score (Gen)'] = nnd_metric_fn(gen_repr_nnd, train_repr_my)
            if index == 'all_samples' or index.startswith('class_'):
                result[index]['Novelty-Score (Test)'] = nnd_metric_fn(ori_repr_nnd, train_repr_my)
                
        if 'LDS' in method_list:
            lds_metric_fn = LatentDensityMetric(sigma=getattr(args, 'sigma', 5.0))
            result[index]['Latent-Density (Gen)'] = lds_metric_fn(gen_repr_nnd, train_repr_my)
            if index == 'all_samples' or index.startswith('class_'):
                result[index]['Latent-Density (Test)'] = lds_metric_fn(ori_repr_nnd, train_repr_my)
    else:
        # Bootstrapping (K-fold Test Split)
        N = ori_repr_fid.shape[0]
        # Only split if we have enough samples (e.g. > 10)
        if N < 10:
            print(f"[{index}] Not enough samples ({N}) for {n_folds}-fold split. Falling back to 1-fold.")
            n_folds = 1
            idx_list = [np.arange(N)]
        else:
            indices = np.random.permutation(N)
            fold_size = N // n_folds
            idx_list = [indices[k * fold_size : (k + 1) * fold_size if k < n_folds - 1 else N] for k in range(n_folds)]
            
        cfid_scores, nnd_g_scores, nnd_t_scores = [], [], []
        lds_g_scores, lds_t_scores = [], []
        
        for k, idx in enumerate(idx_list):
            if 'C-FID' in method_list:
                try:
                    cfid_scores.append(CFIDMetric()(ori_repr_fid[idx], gen_repr_fid[idx], np.array(cond_embs)[idx]))
                except Exception as e:
                    pass
                    
            if 'NND' in method_list:
                try:
                    nnd_metric_fn = NNDMetric()
                    nnd_g_scores.append(nnd_metric_fn(gen_repr_nnd[idx], train_repr_my))
                    if index == 'all_samples' or index.startswith('class_'):
                        nnd_t_scores.append(nnd_metric_fn(ori_repr_nnd[idx], train_repr_my))
                except Exception as e:
                    pass
                    
            if 'LDS' in method_list:
                try:
                    lds_metric_fn = LatentDensityMetric(sigma=getattr(args, 'sigma', 5.0))
                    lds_g_scores.append(lds_metric_fn(gen_repr_nnd[idx], train_repr_my))
                    if index == 'all_samples' or index.startswith('class_'):
                        lds_t_scores.append(lds_metric_fn(ori_repr_nnd[idx], train_repr_my))
                except Exception as e:
                    pass

        # Save means and standard deviations
        if cfid_scores:
            result[index]['C-FID'] = float(np.mean(cfid_scores))
            result[index]['C-FID_std'] = float(np.std(cfid_scores))
        if nnd_g_scores:
            result[index]['Novelty-Score (Gen)'] = float(np.mean(nnd_g_scores))
            result[index]['Novelty-Score (Gen)_std'] = float(np.std(nnd_g_scores))
        if nnd_t_scores:
            result[index]['Novelty-Score (Test)'] = float(np.mean(nnd_t_scores))
            result[index]['Novelty-Score (Test)_std'] = float(np.std(nnd_t_scores))
        if lds_g_scores:
            result[index]['Latent-Density (Gen)'] = float(np.mean(lds_g_scores))
            result[index]['Latent-Density (Gen)_std'] = float(np.std(lds_g_scores))
        if lds_t_scores:
            result[index]['Latent-Density (Test)'] = float(np.mean(lds_t_scores))
            result[index]['Latent-Density (Test)_std'] = float(np.std(lds_t_scores))

    return result, ori_repr_fid, gen_repr_fid
