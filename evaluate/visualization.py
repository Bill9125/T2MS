import os
import glob
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_heatmap(args, root_dir, metric_name, output_name, title_suffix="", group_key="summary"):
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
            # Expected format: .../backbone_denoiser_dataset_cfg_step_subject/...
            parent_dir = os.path.basename(os.path.dirname(fpath))
            parts = parent_dir.split('_')
            # 根據 args.subject 與 args.caption 過濾資料夾
            if f"_{args.subject}_" not in f"_{parent_dir}_":
                continue
            if args.caption not in parent_dir:
                continue
                
            # 尋找 dataset 名稱的位置，其後兩個必定是 cfg 與 step
            if 'benchpress' in parts:
                idx = parts.index('benchpress')
            elif 'deadlift' in parts:
                idx = parts.index('deadlift')
            else:
                continue
                
            cfg = float(parts[idx+1])
            step = int(parts[idx+2])
            
            with open(fpath, 'r') as f:
                content = json.load(f)
                
            if group_key in content and metric_name in content[group_key]:
                val = content[group_key][metric_name]
                data_list.append({'CFG': cfg, 'Step': step, 'Value': val})
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")
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
    save_dir = os.path.join(f'{args.save_path}', 'heatmaps', args.subject, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{output_name}_{exp_type}.png")
    
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def plot_tsne(ori_repr, gen_repr, save_path, title_suffix="", train_repr=None, gen_labels=None, **gen_reprs):
    """
    Plots PCA and t-SNE for Training set, Test set, and one or more Generated sets, with generated classes colored separately.
    """
    if ori_repr is None or (gen_repr is None and not gen_reprs):
        return
        
    # 設定各資料集最大抽樣數，平衡視覺分佈並加速 t-SNE 運算
    max_samples = 2000
    
    if train_repr is not None and len(train_repr) > max_samples:
        indices = np.random.choice(len(train_repr), max_samples, replace=False)
        train_repr = train_repr[indices]
        
    if len(ori_repr) > max_samples:
        indices = np.random.choice(len(ori_repr), max_samples, replace=False)
        ori_repr = ori_repr[indices]
        
    # Extract label lists passed in kwargs
    gen_labels_dict = {}
    for key in list(gen_reprs.keys()):
        if key.endswith("_labels"):
            gen_labels_dict[key[:-7]] = gen_reprs.pop(key)

    # Gather all generated representations and their corresponding labels
    all_gens = {}
    all_gen_labels = {}
    
    if gen_repr is not None:
        if gen_labels is not None:
            labels_arr = np.array(gen_labels)
            # Filter out "unknown" class
            valid_mask = np.array([str(l).lower() != 'unknown' for l in labels_arr])
            if np.any(valid_mask):
                all_gens["Generated set"] = gen_repr[valid_mask]
                all_gen_labels["Generated set"] = labels_arr[valid_mask]
        else:
            all_gens["Generated set"] = gen_repr
            all_gen_labels["Generated set"] = np.array(["Generated set"] * len(gen_repr))
            
    for name, gr in gen_reprs.items():
        if gr is not None:
            if name in gen_labels_dict and gen_labels_dict[name] is not None:
                labels_arr = np.array(gen_labels_dict[name])
                # Filter out "unknown" class
                valid_mask = np.array([str(l).lower() != 'unknown' for l in labels_arr])
                if np.any(valid_mask):
                    all_gens[name] = gr[valid_mask]
                    all_gen_labels[name] = labels_arr[valid_mask]
            else:
                all_gens[name] = gr
                all_gen_labels[name] = np.array([f"Generated ({name})"] * len(gr))
                
    # Sample each generated set and its labels
    for name, gr in list(all_gens.items()):
        if len(gr) > max_samples:
            indices = np.random.choice(len(gr), max_samples, replace=False)
            all_gens[name] = gr[indices]
            all_gen_labels[name] = all_gen_labels[name][indices]

    arrays_to_combine = []
    labels = []
    
    palette = {'Test set': '#3498db'}
    if train_repr is not None:
        arrays_to_combine.append(train_repr)
        labels.extend(['Training set'] * len(train_repr))
        palette['Training set'] = '#cccccc'
        
    arrays_to_combine.append(ori_repr)
    labels.extend(['Test set'] * len(ori_repr))
    
    # Pre-defined colors for different generated classes
    gen_color_idx = 0
    
    for name, gr in all_gens.items():
        arrays_to_combine.append(gr)
        gr_labels = all_gen_labels[name]
        
        # Use cleaner name in the legend if it's "Generated set"
        model_lbl = name if name != "Generated set" else "Generated"
        
        for l in gr_labels:
            labels.append(f"{model_lbl} - {l}")
            
    combined = np.vstack(arrays_to_combine)
    
    # Assign unique colors dynamically to generated labels
    # Use a custom list of vibrant, premium colors that do not conflict with gray/light blue
    premium_colors = [
        '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22',
        '#1abc9c', '#e84393', '#6c5ce7', '#d35400', '#16a085',
        '#8e44ad', '#273c75'
    ]
    unique_labels = sorted(list(set(labels)))
    for lbl in unique_labels:
        if lbl not in palette:
            palette[lbl] = premium_colors[gen_color_idx % len(premium_colors)]
            gen_color_idx += 1
            
    # Order classes to draw Training set in the background, then Test set, then Generated classes
    hue_order = []
    if 'Training set' in unique_labels:
        hue_order.append('Training set')
    if 'Test set' in unique_labels:
        hue_order.append('Test set')
    for lbl in sorted(unique_labels):
        if lbl not in ['Training set', 'Test set']:
            hue_order.append(lbl)
    
    # PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)

    # t-SNE
    n = combined.shape[0]
    perplexity = max(2, min(n - 1, 30))
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto')
    combined_tsne = tsne.fit_transform(combined)

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    sns.scatterplot(x=combined_pca[:, 0], y=combined_pca[:, 1], hue=labels, hue_order=hue_order, ax=axs[0], alpha=0.6, palette=palette, s=40)
    axs[0].set_title(f'PCA Visualization {title_suffix}', fontsize=14)
    axs[0].legend(frameon=True, shadow=True)
    
    sns.scatterplot(x=combined_tsne[:, 0], y=combined_tsne[:, 1], hue=labels, hue_order=hue_order, ax=axs[1], alpha=0.6, palette=palette, s=40)
    axs[1].set_title(f't-SNE Visualization {title_suffix}', fontsize=14)
    axs[1].legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PCA/t-SNE plot saved to: {save_path}")

def generate_heatmaps(args, result):
    evaluation_root = os.path.dirname(args.evaluation_save_path)
    print("\nUpdating Heatmaps...")
    
    plot_heatmap(args, evaluation_root, "C-FID", "heatmap_C_FID_Overall_Average", "(Overall Average)", group_key="summary")
    plot_heatmap(args, evaluation_root, "Novelty-Score (Gen)", "heatmap_Novelty_Gen_Overall_Average", "(Overall Average)", group_key="summary")
    if 'LDS' in args.method_list:
        plot_heatmap(args, evaluation_root, "Latent-Density (Gen)", "heatmap_LDS_Gen_Overall_Average", "(Overall Average)", group_key="summary")
    
    for group_k in result.keys():
        if group_k in ['summary', 'all_samples', 'config']: continue
            
        safe_name = group_k.replace("class_", "")
        plot_heatmap(args, evaluation_root, "C-FID", f"heatmap_C_FID_{safe_name}", f"({safe_name})", group_key=group_k)
        plot_heatmap(args, evaluation_root, "Novelty-Score (Gen)", f"heatmap_Novelty_Gen_{safe_name}", f"({safe_name} vs Train)", group_key=group_k)
        plot_heatmap(args, evaluation_root, "Novelty-Score (Test)", f"heatmap_Novelty_Test_{safe_name}", f"({safe_name} Real vs Train)", group_key=group_k)
        if 'LDS' in args.method_list:
            plot_heatmap(args, evaluation_root, "Latent-Density (Gen)", f"heatmap_LDS_Gen_{safe_name}", f"({safe_name} vs Train)", group_key=group_k)
            plot_heatmap(args, evaluation_root, "Latent-Density (Test)", f"heatmap_LDS_Test_{safe_name}", f"({safe_name} Real vs Train)", group_key=group_k)
