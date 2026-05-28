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

def compute_lds(embs_flat, train_repr_flat, device, sigma=5.0):
    # L2 normalize
    norm_gen = embs_flat / (np.linalg.norm(embs_flat, axis=1, keepdims=True) + 1e-8)
    norm_train = train_repr_flat / (np.linalg.norm(train_repr_flat, axis=1, keepdims=True) + 1e-8)

    Z_g = torch.tensor(norm_gen, dtype=torch.float32, device=device)
    Z_train = torch.tensor(norm_train, dtype=torch.float32, device=device)

    scores = []
    chunk_size = 500
    for idx in range(0, Z_g.shape[0], chunk_size):
        chunk_g = Z_g[idx:idx+chunk_size]
        dist_sq = torch.cdist(chunk_g, Z_train, p=2).pow(2)
        density_matrix = torch.exp(-dist_sq / (2 * sigma ** 2))
        chunk_scores = density_matrix.mean(dim=1).cpu().numpy()
        scores.extend(chunk_scores)
    return np.array(scores)

def main():
    parser = argparse.ArgumentParser(description="LDS Real vs Generated Sample Clustering Experiment")
    parser.add_argument('--dataset_name', '-d', type=str, default='benchpress', help='dataset name (benchpress or deadlift)')
    parser.add_argument('--subject', type=str, default='mix', choices=['isolated', 'mix'], help='subject type')
    parser.add_argument('--cfg_scale', type=int, default=3, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='Total sampling steps')
    parser.add_argument('--top_k', type=int, default=1000, help='Maximum number of samples to select for both Real and Generated groups')
    parser.add_argument('--save_path', type=str, default='./results/clustering_experiment', help='Save path')
    parser.add_argument('--perplexity', type=int, default=None, help='t-SNE perplexity')
    parser.add_argument('--sigma', type=float, default=5.0, help='sigma for Latent Density Score')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for extraction')
    args = parser.parse_args()

    # Dynamic inputs
    args.caption = 'explain'
    args.backbone = 'flowmatching'
    args.denoiser = 'DiT'
    args.method_list = ['TSNE', 'LDS']

    # Load configuration
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Subject mode: {args.subject}")
    print(f"CFG Scale: {args.cfg_scale}, Total Steps: {args.total_step}")
    print(f"Top K samples per class: {args.top_k}")

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

    # Load dataset index (to get all real clips)
    json_path = os.path.join(args.dataset_root, args.dataset_name, 'data.json')
    print(f"Loading real data index from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_real_data = json.load(f)

    # Group real clips by category
    real_data_by_class = {cat: [] for cat in categories}
    target_len = args.split_base_num * 2

    for subj, clips in all_real_data.items():
        main_class = next((cat for cat in categories if cat in subj), None)
        if main_class is None:
            continue # skip unknown classes

        for clip_id, feat_dict in clips.items():
            seqs_T = []
            T_list = []
            for k in args.features:
                if k not in feat_dict:
                    continue
                x = torch.as_tensor(feat_dict[k], dtype=torch.float32)
                if x.dim() == 1:
                    seqs_T.append(x)
                    T_list.append(x.size(0))
            if len(seqs_T) == len(args.features) and len(set(T_list)) == 1:
                x_nfT = torch.stack(seqs_T, dim=0)
                # Align length
                Tcur = x_nfT.size(1)
                Ttar = target_len
                x_1cT = x_nfT.unsqueeze(0)
                if Tcur > Ttar:
                    x_1cT = F.adaptive_avg_pool1d(x_1cT, output_size=Ttar)
                elif Tcur < Ttar:
                    x_1cT = F.interpolate(x_1cT, size=Ttar, mode='linear', align_corners=True)
                real_data_by_class[main_class].append(x_1cT.squeeze(0))

    # Load all generated clips
    run_folders = glob.glob(os.path.join(args.generation_save_path, "run_*"))
    if not run_folders:
        print(f"Error: No generated data found in {args.generation_save_path}")
        return

    generated_data_by_class = {cat: [] for cat in categories}
    for run_path in sorted(run_folders, key=lambda x: int(os.path.basename(x).split('_')[-1])):
        for sample_dir in os.listdir(run_path):
            sample_path = os.path.join(run_path, sample_dir)
            if not os.path.isdir(sample_path) or sample_dir.startswith('.'):
                continue
                
            x_t_path = os.path.join(sample_path, 'x_t.npy')
            if os.path.exists(x_t_path):
                error_class = next((k for k in categories if k in sample_dir), None)
                if error_class is None:
                    continue
                x_t = np.load(x_t_path)
                generated_data_by_class[error_class].append(x_t)

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    print("\n--- Running Feature Extraction & LDS Filtering ---")
    for cat in categories:
        print(f"\n==========================================")
        print(f"Analyzing Category: [{cat}]")
        print(f"==========================================")

        real_clips = real_data_by_class[cat]
        gen_clips = generated_data_by_class[cat]

        if not real_clips or not gen_clips:
            print(f"Warning: Missing data for {cat}. Real clips: {len(real_clips)}, Gen clips: {len(gen_clips)}")
            continue

        # Extract features for Real clips
        print(f" - Extracting features for {len(real_clips)} REAL clips...")
        real_embs_mean_list = []
        real_embs_flat_list = []
        for i in range(0, len(real_clips), args.batch_size):
            batch = torch.stack(real_clips[i:i+args.batch_size]).to(args.device)
            with torch.no_grad():
                z, _ = vae_encoder(batch)
            real_embs_mean_list.append(z.mean(dim=-1).cpu().numpy())
            real_embs_flat_list.append(z.flatten(start_dim=1).cpu().numpy())
        real_mean = np.concatenate(real_embs_mean_list, axis=0)
        real_flat = np.concatenate(real_embs_flat_list, axis=0)

        # Extract features for Gen clips
        print(f" - Extracting features for {len(gen_clips)} GENERATED clips...")
        gen_embs_mean_list = []
        gen_embs_flat_list = []
        padded_gen = []
        for x in gen_clips:
            x_nfT = torch.as_tensor(x, dtype=torch.float32)
            Tcur = x_nfT.size(1)
            Ttar = target_len
            x_1cT = x_nfT.unsqueeze(0)
            if Tcur > Ttar:
                x_1cT = F.adaptive_avg_pool1d(x_1cT, output_size=Ttar)
            elif Tcur < Ttar:
                x_1cT = F.interpolate(x_1cT, size=Ttar, mode='linear', align_corners=True)
            padded_gen.append(x_1cT.squeeze(0))

        for i in range(0, len(padded_gen), args.batch_size):
            batch = torch.stack(padded_gen[i:i+args.batch_size]).to(args.device)
            with torch.no_grad():
                z, _ = vae_encoder(batch)
            gen_embs_mean_list.append(z.mean(dim=-1).cpu().numpy())
            gen_embs_flat_list.append(z.flatten(start_dim=1).cpu().numpy())
        gen_mean = np.concatenate(gen_embs_mean_list, axis=0)
        gen_flat = np.concatenate(gen_embs_flat_list, axis=0)

        # Compute LDS scores
        real_scores = compute_lds(real_flat, train_repr_flat, args.device, args.sigma)
        gen_scores = compute_lds(gen_flat, train_repr_flat, args.device, args.sigma)

        # Determine balanced sample count K
        K = min(len(real_clips), len(gen_clips), args.top_k)

        # Sort and select Top K
        real_top_idx = np.argsort(real_scores)[::-1][:K]
        gen_top_idx = np.argsort(gen_scores)[::-1][:K]

        sel_real_mean = real_mean[real_top_idx]
        sel_real_flat = real_flat[real_top_idx]
        sel_gen_mean = gen_mean[gen_top_idx]
        sel_gen_flat = gen_flat[gen_top_idx]

        print(f" - Selected exactly {K} Real and {K} Generated clips (1:1 balanced ratio)!")
        print(f"   (Real Avg LDS: {real_scores[real_top_idx].mean():.4f} | Gen Avg LDS: {gen_scores[gen_top_idx].mean():.4f})")

        # Run dimensionality reduction for both Embedding Types
        for name, r_emb, g_emb in [("Mean-pooled", sel_real_mean, sel_gen_mean), ("Flattened", sel_real_flat, sel_gen_flat)]:
            combined = np.vstack([r_emb, g_emb])
            labels = ["Real"] * len(r_emb) + ["Generated"] * len(g_emb)

            # Compute Silhouette Score between Real and Generated
            overlap_score = silhouette_score(combined, labels)
            print(f"   [{name}] Silhouette Score (Real vs Gen): {overlap_score:.4f} (lower means closer distributions)")

            # PCA
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined)

            # t-SNE
            n = combined.shape[0]
            if args.perplexity is not None:
                perplexity = max(2, min(n - 1, args.perplexity))
            else:
                if n > 1000:
                    perplexity = 100
                else:
                    perplexity = max(2, min(n - 1, 30))

            tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=args.general_seed)
            combined_tsne = tsne.fit_transform(combined)

            # Create plots
            fig, axs = plt.subplots(1, 2, figsize=(18, 8))
            palette = {"Real": "#3498db", "Generated": "#e74c3c"} # Blue and Red

            sns.scatterplot(x=combined_pca[:, 0], y=combined_pca[:, 1], hue=labels, ax=axs[0], alpha=0.6, palette=palette, s=40)
            axs[0].set_title(f'PCA - Real vs Generated [{cat}]\n(Overlap Silhouette: {overlap_score:.4f})', fontsize=14)
            axs[0].legend(frameon=True, shadow=True)
            axs[0].grid(True, linestyle='--', alpha=0.5)

            sns.scatterplot(x=combined_tsne[:, 0], y=combined_tsne[:, 1], hue=labels, ax=axs[1], alpha=0.6, palette=palette, s=40)
            axs[1].set_title(f't-SNE - Real vs Generated [{cat}]\n(Perplexity: {perplexity})', fontsize=14)
            axs[1].legend(frameon=True, shadow=True)
            axs[1].grid(True, linestyle='--', alpha=0.5)

            plt.suptitle(f'{name} Embeddings (Real: {len(r_emb)} clips, Gen: {len(g_emb)} clips) - {args.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            plot_filename = f"real_vs_gen_{cat}_{name.lower().replace('-', '_')}_{args.dataset_name}.png"
            plot_save_path = os.path.join(args.save_path, plot_filename)
            plt.savefig(plot_save_path, dpi=150)
            plt.close()
            print(f"   Saved plot: {plot_save_path}")

    print(f"\nReal vs Generated Clustering Experiment for '{args.dataset_name}' Completed successfully!")
    print(f"Plots saved to: {os.path.abspath(args.save_path)}")

if __name__ == '__main__':
    main()
