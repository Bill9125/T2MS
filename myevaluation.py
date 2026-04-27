import os
import datetime
import json
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import torch
from evaluate.utils import show_with_start_divider, show_with_end_divider, write_json_data
from utils import get_cfg
from model.pretrained.myvqvae import vqvae
from evaluate.cfid import CFIDMetric
from evaluate.nnd import NNDMetric

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def normalize(x):
    # Normalize along the time dimension for each channel
    min_val = x.min(axis=1, keepdims=True)
    max_val = x.max(axis=1, keepdims=True)
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    return x_norm



def plot_heatmap(root_dir, metric_name, output_name, title_suffix="", group_key="summary"):
    """
    Collects JSON files from specified directory, extracts CFG and Step, 
    and plots a heatmap for the specified metric under a specific group_key (e.g., 'class_tilting_to_the_left').
    """
    data_list = []
    # Find all JSON evaluation files (keeping only the latest for each combination)
    all_json_files = glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True)
    
    if not all_json_files:
        print(f"No JSON files found in {root_dir}")
        return
        
    latest_jsons = {}
    for fpath in all_json_files:
        pdir = os.path.dirname(fpath)
        if pdir not in latest_jsons or os.path.getmtime(fpath) > os.path.getmtime(latest_jsons[pdir]):
            latest_jsons[pdir] = fpath

    for fpath in latest_jsons.values():
        try:
            # Expected format: .../backbone_denoiser_dataset_cfg_step/...
            parent_dir = os.path.basename(os.path.dirname(fpath))
            parts = parent_dir.split('_')
            cfg = float(parts[-2])
            step = int(parts[-1])
            
            with open(fpath, 'r') as f:
                content = json.load(f)
                
            if group_key in content and metric_name in content[group_key]:
                val = content[group_key][metric_name]
                data_list.append({'CFG': cfg, 'Step': step, 'Value': val})
        except Exception:
            continue

    if not data_list:
        print(f"Metric {metric_name} not found in JSON results.")
        return

    df = pd.DataFrame(data_list)
    pivot_table = df.pivot_table(index='Step', columns='CFG', values='Value')
    pivot_table = pivot_table.sort_index(ascending=False) # High step counts at the top

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu_r") 
    plt.title(f'Heatmap of {metric_name} {title_suffix}')
    plt.xlabel('CFG Scale')
    plt.ylabel('Total Steps')
    # 透過 root_dir 來判別這是 fixed 還是 random 的實驗 (e.g., evaluation_fixed)
    exp_type = os.path.basename(root_dir).replace("evaluation_", "")
    
    # 統一儲存到 ./heatmaps 資料夾底下
    save_dir = os.path.join('.', 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{output_name}_{exp_type}.png")
    
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

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
        return None
    
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

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate flow matching model")
    parser.add_argument('--method_list', type=str, default='C-FID,NND',
                            help='metric list [C-FID, NND]')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Save path')
    parser.add_argument('--config', type=str, default='config.yaml', help='configuration file')
    parser.add_argument('--dataset_name', '-d', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--cfg_scale', type=int, default=1, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='Total sampling steps')
    parser.add_argument('--run_time', type=int, default=10, help='Number of runs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--fixed_noise', action='store_true', help='Evaluate fixed random noise dataset')
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds to split the evaluation results to get standard deviation')

    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load VAE model
    args.pretrainedvae_path = os.path.join('./results/saved_pretrained_models', 
                                         f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}', 
                                         'final_model.pth')
    print('Loading pretrained VAE encoder from: ', args.pretrainedvae_path)
    vae = vqvae(args).to(args.device).float().eval()
    vae.load_state_dict(torch.load(args.pretrainedvae_path, map_location=args.device))
    vae_encoder = vae.encoder

    args.model_name = f'{args.backbone}_{args.denoiser}_{args.dataset_name}_{args.cfg_scale}_{args.total_step}'
    
    noise_type = "fixed" if getattr(args, 'fixed_noise', False) else "random"
    args.generation_save_path = os.path.join(args.save_path, f'generation_{noise_type}', args.model_name)
    args.evaluation_save_path = os.path.join(args.save_path, f'evaluation_{noise_type}', args.model_name)

    # Load training data for Novelty (NND) calculation
    train_repr_my = None
    if 'NND' in args.method_list:
        if args.dataset_name == 'benchpress':
            from datafactory.benchpress.dataloader import loader_provider
        elif args.dataset_name == 'deadlift':
            from datafactory.deadlift.dataloader import loader_provider
        
        train_loader, _ = loader_provider(args, period='train')
        print(f"Loading training data for Novelty calculation... (Total batches: {len(train_loader)})")
        
        train_repr_list = []
        train_labels_list = []
        train_embs_list = []
        with torch.no_grad():
            for batch_data in train_loader:
                if isinstance(batch_data, list):
                    texts, xs, embs, subs, clips = batch_data[0]
                else:
                    texts, xs, embs, subs, clips = batch_data
                
                xs = xs.float().to(args.device)
                features, _ = vae_encoder(xs)
                # For NND, use flattened features
                repr_nnd = features.flatten(start_dim=1).cpu().numpy()
                train_repr_list.append(repr_nnd)
                
                # 收集文字 Embedding 用於 CFID
                train_embs_list.append(embs.cpu().numpy())
                
                # 收集每一筆訓練資料的錯誤標籤 (從 subject 字串中獲取)
                for s in subs:
                    s_str = s[0] if isinstance(s, tuple) else s
                    train_labels_list.append(str(s_str))
        
        train_repr_my = np.concatenate(train_repr_list, axis=0)
        train_labels_my = np.array(train_labels_list)
        train_embs_my = np.concatenate(train_embs_list, axis=0)
        print(f"Loaded {len(train_labels_my)} training labels.")
        print(f"Sample training labels: {train_labels_my[:10]}")
    else:
        train_repr_my = None
        train_labels_my = None
        train_embs_my = None


    result = {}
    x_1_list = []
    x_t_list = []
    emb_list = [] # 新增 Embedding 列表

    # 準備一個 dictionary 來將樣本按照類別 (error label) 進行分組
    grouped_samples = {}

    # Gather generated and original samples
    for j in range(args.run_time):
        run_save_path = os.path.join(args.generation_save_path, f'run_{j}')
        if not os.path.exists(run_save_path):
            continue
            
        for sample_dir in os.listdir(run_save_path):
            sample_path = os.path.join(run_save_path, sample_dir)
            if not os.path.isdir(sample_path) or sample_dir.startswith('.'):
                continue
                
            x_t_path = os.path.join(sample_path, 'x_t.npy')
            x_1_path = os.path.join(sample_path, 'x_1.npy')
            emb_path = os.path.join(sample_path, 'embedding.npy')
            
            if os.path.exists(x_t_path) and os.path.exists(x_1_path):
                x_t = normalize(np.load(x_t_path))
                x_1 = normalize(np.load(x_1_path))
                
                # 載入 Embedding
                if os.path.exists(emb_path):
                    emb = np.load(emb_path)
                    if emb.ndim == 1:
                        emb = np.expand_dims(emb, axis=0) # [1, Dim]
                    if emb.ndim == 3: # 防呆
                        emb = emb.squeeze(1)
                else:
                    emb = np.zeros((1, args.embedding_dim))
                
                x_t_list.append(x_t)
                x_1_list.append(x_1)
                emb_list.append(emb) # 記錄到全局列表
                
                # 從資料夾名稱解析 error class
                known_classes = [
                    'tilting_to_the_left', 
                    'tilting_to_the_right', 
                    'scapular_protraction', 
                    'elbows_flaring',
                ]
                
                error_class = None
                for k_class in known_classes:
                    if k_class in sample_dir:
                        error_class = k_class
                        break
                
                # 若不在目標分類中，直接忽略個別分類評估
                if error_class is None:
                    continue

                if error_class not in grouped_samples:
                    grouped_samples[error_class] = {'x_1': [], 'x_t': [], 'emb': []}
                
                grouped_samples[error_class]['x_1'].append(x_1)
                grouped_samples[error_class]['x_t'].append(x_t)
                grouped_samples[error_class]['emb'].append(emb)

    if x_t_list:
        # Align lengths with zero padding if necessary
        max_len = max(x.shape[-1] for x in x_t_list)
        
        # 1. 跑全域評估 (Global Marginals)
        ori_data_arr = np.array([np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), 'constant') for x in x_1_list])
        gen_data_arr = np.array([np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), 'constant') for x in x_t_list])
        all_embs_arr = np.concatenate(emb_list, axis=0)
        
        print(f'Original data shape: {ori_data_arr.shape}, Generated data shape: {gen_data_arr.shape}')
        result = evaluate_data(args, ori_data_arr, gen_data_arr, 'all_samples', result, vae_encoder=vae_encoder, train_repr_my=train_repr_my, cond_embs=all_embs_arr)
        
        # 2. 跑各個分類的獨立評估 (Class-Conditional Evaluation)
        print("\n--- Running Class-Conditional Evaluation ---")
        for error_class, data_dict in grouped_samples.items():
            if len(data_dict['x_t']) <= 1:
                print(f"Skipping class [{error_class}] - Not enough samples for covariance calculation.")
                continue
                
            print(f"Evaluating Class: [{error_class}] ({len(data_dict['x_t'])} samples)")
            class_max_len = max(x.shape[-1] for x in data_dict['x_t'])
            class_ori_arr = np.array([np.pad(x, ((0, 0), (0, class_max_len - x.shape[-1])), 'constant') for x in data_dict['x_1']])
            class_gen_arr = np.array([np.pad(x, ((0, 0), (0, class_max_len - x.shape[-1])), 'constant') for x in data_dict['x_t']])
            class_embs_arr = np.concatenate(data_dict['emb'], axis=0)
            
            class_train_repr = None
            if train_repr_my is not None:
                # 只篩選屬於這個錯誤類別的訓練集特徵來作為基準
                # 改為用「包含」來判定，因為訓練集標籤通常也是包含 subject_ID 等資訊的長字串
                mask = np.array([error_class in str(label) for label in train_labels_my])
                if np.sum(mask) > 0:
                    class_train_repr = train_repr_my[mask]
                    class_train_embs = train_embs_my[mask] 
                else:
                    print(f"Warning: No training samples found matching class '{error_class}'. Using full train set as fallback.")
                    class_train_repr = train_repr_my
                    class_train_embs = train_embs_my
                    
            # 傳入條件文字向量進行 CFID 計算
            result = evaluate_data(args, class_ori_arr, class_gen_arr, f'class_{error_class}', result, vae_encoder=vae_encoder, train_repr_my=class_train_repr, cond_embs=class_embs_arr)
        print("--------------------------------------------\n")

    if isinstance(result, dict) and result:
        # Calculate summary
        summary = {}
        for key in result:
            for metric, value in result[key].items():
                summary[metric] = summary.get(metric, 0) + value
        for metric in summary:
            summary[metric] = (summary[metric] / len(result))
            
        result['summary'] = summary
        result = convert_numpy(result)
        
        os.makedirs(args.evaluation_save_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(args.evaluation_save_path, f'{args.model_name}_{timestamp}.json')
        write_json_data(result, save_path)
        print(f'Evaluation results saved to {save_path}')

        # Generate Heatmaps automatically if evaluation was successful
        evaluation_root = os.path.dirname(args.evaluation_save_path)
        print("\nUpdating Heatmaps...")
        
        # 1. 繪製「加總平均」與「不分類大雜燴」版本
        plot_heatmap(evaluation_root, "C-FID", "heatmap_C_FID_Overall_Average", "(Overall Average)", group_key="summary")
        plot_heatmap(evaluation_root, "Novelty-Score (Gen)", "heatmap_Novelty_Gen_Overall_Average", "(Overall Average)", group_key="summary")
        
        # 2. 針對「每一種錯誤類別」各自畫圖
        for group_k in result.keys():
            if group_k in ['summary', 'all_samples']:
                continue
                
            safe_name = group_k.replace("class_", "")
            plot_heatmap(evaluation_root, "C-FID", f"heatmap_C_FID_{safe_name}", f"({safe_name})", group_key=group_k)
            plot_heatmap(evaluation_root, "Novelty-Score (Gen)", f"heatmap_Novelty_Gen_{safe_name}", f"({safe_name} vs Train)", group_key=group_k)
            plot_heatmap(evaluation_root, "Novelty-Score (Test)", f"heatmap_Novelty_Test_{safe_name}", f"({safe_name} Real vs Train)", group_key=group_k)
    
    show_with_end_divider(f'Evaluation done. Results: {result}')
