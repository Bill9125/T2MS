import os
import glob
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from utils import get_cfg
from model.pretrained.myvqvae import vqvae
from evaluate.data import load_training_data

def load_vae_model(args):
    print('Loading pretrained VAE encoder from: ', args.pretrainedvae_path)
    vae = vqvae(args).to(args.device).float().eval()
    vae.load_state_dict(torch.load(args.pretrainedvae_path, map_location=args.device))
    return vae.encoder

def main():
    parser = argparse.ArgumentParser(description="LDS Filtered Generated Sample Clustering Experiment")
    parser.add_argument('--dataset_name', '-d', type=str, default='benchpress', help='dataset name (benchpress or deadlift)')
    parser.add_argument('--subject', type=str, default='mix', choices=['isolated', 'mix'], help='subject type')
    parser.add_argument('--cfg_scale', type=int, default=1, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='Total sampling steps')
    parser.add_argument('--top_k', type=int, default=2000, help='Number of top LDS samples to select per class')
    parser.add_argument('--save_path', type=str, default='./results/clustering_experiment', help='Save path')
    parser.add_argument('--perplexity', type=int, default=None, help='t-SNE perplexity')
    parser.add_argument('--sigma', type=float, default=5.0, help='sigma for Latent Density Score')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for extraction')
    args = parser.parse_args()

    # Dynamic inputs
    args.caption = 'explain' # Default evaluation caption
    args.backbone = 'flowmatching'
    args.denoiser = 'DiT'
    args.method_list = ['TSNE', 'LDS'] # required to load training data

    # Load configuration
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Subject mode: {args.subject}")
    print(f"CFG Scale: {args.cfg_scale}, Total Steps: {args.total_step}")
    print(f"Top K per category: {args.top_k}")

    # Set VAE and generation folders paths
    args.pretrainedvae_path = os.path.join(
        './results/saved_pretrained_models', 
        f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}', 
        'final_model.pth'
    )
    args.model_name = f'{args.backbone}_{args.denoiser}_{args.dataset_name}_{args.cfg_scale}_{args.total_step}_{args.subject}_{args.caption}'
    args.generation_save_path = os.path.join('./results/denoiser_results/generation_random', args.model_name)

    print(f"VAE checkpoint path: {args.pretrainedvae_path}")
    print(f"Generation save path: {args.generation_save_path}")

    # Load VAE Encoder
    vae_encoder = load_vae_model(args)

    # Determine movement categories based on dataset
    if args.dataset_name == 'benchpress':
        categories = ['correct', 'tilting_to_the_left', 'tilting_to_the_right', 'scapular_protraction', 'elbows_flaring']
    elif args.dataset_name == 'deadlift':
        categories = ['Correct', 'Barbell_moving_away_from_the_shins', 'Barbell_colliding_with_the_knees', 'Lower_back_rounding', 'Hips_rising_before_the_barbell_leaves_the_ground']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Load real training dataset representations
    print("\nLoading real training representations for LDS calculation...")
    train_repr_flat, _, _, _ = load_training_data(args, vae_encoder)
    print(f"Loaded training representation shape: {train_repr_flat.shape}")

    # Load all generated clips
    run_folders = glob.glob(os.path.join(args.generation_save_path, "run_*"))
    if not run_folders:
        print(f"Error: No generated data found in {args.generation_save_path}")
        return

    print(f"Found {len(run_folders)} run folders.")
    
    generated_data_by_class = {cat: [] for cat in categories}
    target_len = args.split_base_num * 2

    # Collect all generated samples and group by class
    print("Collecting generated clips...")
    for run_path in sorted(run_folders, key=lambda x: int(os.path.basename(x).split('_')[-1])):
        for sample_dir in os.listdir(run_path):
            sample_path = os.path.join(run_path, sample_dir)
            if not os.path.isdir(sample_path) or sample_dir.startswith('.'):
                continue
                
            x_t_path = os.path.join(sample_path, 'x_t.npy')
            if os.path.exists(x_t_path):
                # Identify class
                error_class = next((k for k in categories if k in sample_dir), None)
                if error_class is None:
                    continue # Skip unknown classes like wrist_bending_backward

                x_t = np.load(x_t_path)
                generated_data_by_class[error_class].append(x_t)

    # Report count per class
    print("\nGenerated clips counts:")
    for cat in categories:
        print(f" - {cat}: {len(generated_data_by_class[cat])} clips")

    # Batched VAE extraction and LDS calculation for each class
    selected_clips_mean_pooled = []
    selected_clips_flattened = []
    selected_labels_class = []

    print("\n--- Running Batched Feature Extraction & LDS Selection ---")
    for cat in categories:
        clips = generated_data_by_class[cat]
        if not clips:
            continue
            
        print(f"Processing Category: {cat}...")
        
        # 1. Pad and stack to tensor
        padded_clips = []
        for x in clips:
            x_nfT = torch.as_tensor(x, dtype=torch.float32)
            Tcur = x_nfT.size(1)
            Ttar = target_len
            x_1cT = x_nfT.unsqueeze(0)
            if Tcur > Ttar:
                x_1cT = F.adaptive_avg_pool1d(x_1cT, output_size=Ttar)
            elif Tcur < Ttar:
                x_1cT = F.interpolate(x_1cT, size=Ttar, mode='linear', align_corners=True)
            padded_clips.append(x_1cT.squeeze(0))

        # 2. Batched feature extraction
        embs_mean_list = []
        embs_flat_list = []
        
        for i in range(0, len(padded_clips), args.batch_size):
            batch = torch.stack(padded_clips[i:i+args.batch_size]).to(args.device)
            with torch.no_grad():
                z, _ = vae_encoder(batch) # [B, embedding_dim, flow_dim]
            
            embs_mean_list.append(z.mean(dim=-1).cpu().numpy())
            embs_flat_list.append(z.flatten(start_dim=1).cpu().numpy())

        cat_embs_mean = np.concatenate(embs_mean_list, axis=0)
        cat_embs_flat = np.concatenate(embs_flat_list, axis=0)

        # 3. Compute LDS for each sample against training representation
        # L2 normalize
        norm_gen = cat_embs_flat / (np.linalg.norm(cat_embs_flat, axis=1, keepdims=True) + 1e-8)
        norm_train = train_repr_flat / (np.linalg.norm(train_repr_flat, axis=1, keepdims=True) + 1e-8)

        Z_g = torch.tensor(norm_gen, dtype=torch.float32, device=args.device)
        Z_train = torch.tensor(norm_train, dtype=torch.float32, device=args.device)

        scores = []
        # Chunk distances to prevent GPU out of memory
        chunk_size = 500
        for idx in range(0, Z_g.shape[0], chunk_size):
            chunk_g = Z_g[idx:idx+chunk_size]
            dist_sq = torch.cdist(chunk_g, Z_train, p=2).pow(2)
            density_matrix = torch.exp(-dist_sq / (2 * args.sigma ** 2))
            chunk_scores = density_matrix.mean(dim=1).cpu().numpy()
            scores.extend(chunk_scores)

        scores = np.array(scores)

        # 4. Sort and pick top K
        sorted_indices = np.argsort(scores)[::-1] # descending order (highest score first)
        top_indices = sorted_indices[:args.top_k]
        
        print(f" - Average LDS for ALL generated samples: {scores.mean():.4f}")
        print(f" - Average LDS for TOP {len(top_indices)} selected samples: {scores[top_indices].mean():.4f}")

        selected_clips_mean_pooled.append(cat_embs_mean[top_indices])
        selected_clips_flattened.append(cat_embs_flat[top_indices])
        selected_labels_class.extend([cat] * len(top_indices))

    # Combine selected samples across all classes
    selected_mean_pooled = np.concatenate(selected_clips_mean_pooled, axis=0)
    selected_flattened = np.concatenate(selected_clips_flattened, axis=0)
    
    print(f"\nFinal Selected Dataset Size: {selected_mean_pooled.shape[0]} clips")

    # 5. Perform PCA & t-SNE Clustering on the high-fidelity LDS subset
    os.makedirs(args.save_path, exist_ok=True)

    for emb_name, embs in [("Mean-pooled", selected_mean_pooled), ("Flattened", selected_flattened)]:
        print(f"\n--- Clustering Analysis for {emb_name} LDS-Filtered Embeddings ---")

        class_silhouette = silhouette_score(embs, selected_labels_class)
        print(f"Silhouette Score (by Movement Class): {class_silhouette:.4f}")

        # Core dimensionality reduction code matches evaluate/visualization.py:L146-L164 exactly
        # PCA
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(embs)

        # t-SNE
        n = embs.shape[0]
        if args.perplexity is not None:
            perplexity = max(2, min(n - 1, args.perplexity))
        else:
            if n > 1000:
                perplexity = 100
            else:
                perplexity = max(2, min(n - 1, 30))

        tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=args.general_seed)
        combined_tsne = tsne.fit_transform(embs)

        # Plot Class clustering
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        class_palette = sns.color_palette("Set1", len(categories))
        
        sns.scatterplot(x=combined_pca[:, 0], y=combined_pca[:, 1], hue=selected_labels_class, ax=axs[0], alpha=0.8, palette=class_palette, s=40)
        axs[0].set_title(f'PCA - Colored by Movement Class\n(Silhouette: {class_silhouette:.4f})', fontsize=14)
        axs[0].legend(frameon=True, shadow=True, title="Movement Class")
        axs[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(x=combined_tsne[:, 0], y=combined_tsne[:, 1], hue=selected_labels_class, ax=axs[1], alpha=0.8, palette=class_palette, s=40)
        axs[1].set_title(f't-SNE - Colored by Movement Class\n(Perplexity: {perplexity})', fontsize=14)
        axs[1].legend(frameon=True, shadow=True, title="Movement Class")
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f'{emb_name} Embeddings (LDS Top {args.top_k} per class) - {args.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_save_path = os.path.join(args.save_path, f"lds_filtered_by_class_{emb_name.lower().replace('-', '_')}_{args.dataset_name}.png")
        plt.savefig(plot_save_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_save_path}")

    print(f"\nLDS Clustering Experiment for '{args.dataset_name}' Completed successfully!")
    print(f"Plots saved to: {os.path.abspath(args.save_path)}")

if __name__ == '__main__':
    main()
