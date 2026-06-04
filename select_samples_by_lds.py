"""
select_samples_by_lds.py

Filters generated samples based on Latent Density Score (LDS) calculated against
the training data distribution.

For each movement category, it selects:
  - Top-6 (highest LDS scores - closest to the training distribution)
  - Mid-6 (median LDS scores - average quality)
  - Bottom-6 (lowest LDS scores - outliers/most diverse or lower quality)

Saves the selected sample names and scores in JSON, and generates grid plots
of their multi-feature trajectories.

Usage:
    # CLIP model
    conda run -n T2S python select_samples_by_lds.py -d benchpress --subject mix --use_clip --cfg_scale 3 --total_step 100

    # Legacy model
    conda run -n T2S python select_samples_by_lds.py -d benchpress --subject mix --cfg_scale 3 --total_step 100
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import get_cfg
from model.pretrained.myvqvae import vqvae
from evaluate.data import load_training_data
from utils.visualize import RearV_BenchpressAnimator, TopV_BenchpressAnimator, LateralV_BenchpressAnimator


def load_vae_model(args):
    print(f"Loading pretrained VAE encoder from: {args.pretrainedvae_path}")
    vae = vqvae(args).to(args.device).float().eval()
    vae.load_state_dict(torch.load(args.pretrainedvae_path, map_location=args.device, weights_only=False))
    return vae.encoder


def plot_trajectories(selected_data, category, group_name, save_dir, features):
    """
    Plots a 2x3 grid of the 6 selected samples' trajectories.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    # Use distinct colors for features
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(features))]
    
    for idx in range(6):
        ax = axs[idx]
        if idx < len(selected_data):
            item = selected_data[idx]
            x_t = item['x_t']
            score = item['score']
            name = item['sample_name']
            
            for f_idx in range(x_t.shape[0]):
                ax.plot(x_t[f_idx], label=features[f_idx] if (idx == 0 and f_idx < 10) else "", color=colors[f_idx], lw=1.2)
            
            # Simple title with shortened name
            short_name = "_".join(name.split("_")[:3] + name.split("_")[-2:])
            ax.set_title(f"{short_name}\nLDS Score: {score:.5f}", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No Sample", ha='center', va='center')
            
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Place legend on the side
    fig.legend(loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=7, ncol=1)
    plt.suptitle(f"{category.upper()} - {group_name} Trajectories (LDS Filtered)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save path
    filename = f"lds_{category}_{group_name.lower().replace('-', '_')}.png"
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Saved Plot] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="LDS Top-6 / Mid-6 / Bottom-6 Sample Filter")
    parser.add_argument('--dataset_name', '-d', type=str, required=True, choices=['deadlift', 'benchpress'], help='dataset name')
    parser.add_argument('--subject', type=str, default='mix', choices=['isolated', 'mix'], help='subject type')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='Total sampling steps')
    parser.add_argument('--caption', type=str, default='explain', choices=['explain', 'style_new'], help='caption type')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='denoiser results save path')
    parser.add_argument('--use_clip', action='store_true', help='use CLIP model')
    parser.add_argument('--pretrainedvae_path', type=str, default=None, help='pretrained VAE path')
    parser.add_argument('--run_id', type=int, default=0, help='Run ID')
    parser.add_argument('--sigma', type=float, default=0.2, help='sigma for Latent Density Score')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for extraction')
    parser.add_argument('--generate_gifs', action='store_true', help='generate skeleton animation GIFs for all selected samples')
    args = parser.parse_args()

    # Fixed configurations
    args.backbone = 'flowmatching'
    args.denoiser = 'DiT'
    args.method_list = ['LDS'] # for load_training_data
    
    # Parse YAML configuration
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clip_tag = 'clip' if args.use_clip else 'legacy'
    
    # Format cfg_scale as integer if it represents a whole number (to match save directory names)
    cfg_str = str(int(args.cfg_scale)) if args.cfg_scale.is_integer() else str(args.cfg_scale)

    # Set paths
    if not args.pretrainedvae_path:
        vae_dir = f'clip_{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}' if args.use_clip else f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}'
        args.pretrainedvae_path = os.path.join('./results/saved_pretrained_models', vae_dir, 'final_model.pth')
        
    args.model_name = f'{args.backbone}_{args.denoiser}_{args.dataset_name}_{cfg_str}_{args.total_step}_{args.subject}_{args.caption}'
    args.generation_save_path = os.path.join(args.save_path, 'generation_random', args.model_name, f'run_{args.run_id}')

    print(f"\n--- LDS Filter Configuration ---")
    print(f"Device:               {args.device}")
    print(f"VAE Path:             {args.pretrainedvae_path}")
    print(f"Generation Dir:       {args.generation_save_path}")
    print(f"LDS Sigma:            {args.sigma}")

    if not os.path.exists(args.generation_save_path):
        raise FileNotFoundError(f"Generation path does not exist: {args.generation_save_path}")

    # Load VAE Encoder
    vae_encoder = load_vae_model(args)

    # Determine movement categories
    if args.dataset_name == 'benchpress':
        categories = ['correct', 'tilting_to_the_left', 'tilting_to_the_right', 'scapular_protraction', 'elbows_flaring']
    else:
        categories = ['Correct', 'Barbell_moving_away_from_the_shins', 'Barbell_colliding_with_the_knees', 'Lower_back_rounding', 'Hips_rising_before_the_barbell_leaves_the_ground']

    # Load training representations
    print("\nLoading training representations for LDS reference...")
    train_repr_flat, _, _, _ = load_training_data(args, vae_encoder)
    print(f"Loaded training representation shape: {train_repr_flat.shape}")

    # Read generated samples grouped by category
    print("\nScanning generated samples...")
    grouped_samples = {cat: [] for cat in categories}
    target_len = args.split_base_num * 2

    for sample_dir in os.listdir(args.generation_save_path):
        sample_path = os.path.join(args.generation_save_path, sample_dir)
        if not os.path.isdir(sample_path) or sample_dir.startswith('.'):
            continue
            
        x_t_path = os.path.join(sample_path, 'x_t.npy')
        if os.path.exists(x_t_path):
            error_class = next((k for k in categories if k in sample_dir), None)
            if error_class is None:
                continue # Skip unrelated mistake classes (e.g. wrist bending)
                
            x_t = np.load(x_t_path)
            grouped_samples[error_class].append({
                'sample_name': sample_dir,
                'sample_path': sample_path,
                'x_t': x_t
            })

    for cat in categories:
        print(f" - {cat}: {len(grouped_samples[cat])} samples found.")

    output_results = {}
    output_dir = os.path.join(args.save_path, "lds_filtering", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Running Feature Extraction & Filtering ---")
    for cat in categories:
        samples = grouped_samples[cat]
        if len(samples) < 6:
            print(f"\n[Warning] Category '{cat}' has only {len(samples)} samples. Skipping selection.")
            continue
            
        print(f"\nProcessing Category: {cat}...")

        # 1. Pad and stack to tensor
        padded_clips = []
        for item in samples:
            x_nfT = torch.as_tensor(item['x_t'], dtype=torch.float32)
            Tcur = x_nfT.size(1)
            Ttar = target_len
            x_1cT = x_nfT.unsqueeze(0)
            if Tcur > Ttar:
                x_1cT = F.adaptive_avg_pool1d(x_1cT, output_size=Ttar)
            elif Tcur < Ttar:
                x_1cT = F.interpolate(x_1cT, size=Ttar, mode='linear', align_corners=True)
            padded_clips.append(x_1cT.squeeze(0))

        # 2. Extract features
        embs_flat_list = []
        for i in range(0, len(padded_clips), args.batch_size):
            batch = torch.stack(padded_clips[i:i+args.batch_size]).to(args.device)
            with torch.no_grad():
                z, _ = vae_encoder(batch)
            embs_flat_list.append(z.flatten(start_dim=1).cpu().numpy())

        cat_embs_flat = np.concatenate(embs_flat_list, axis=0)

        # 3. Calculate LDS scores
        norm_gen = cat_embs_flat / (np.linalg.norm(cat_embs_flat, axis=1, keepdims=True) + 1e-8)
        norm_train = train_repr_flat / (np.linalg.norm(train_repr_flat, axis=1, keepdims=True) + 1e-8)

        Z_g = torch.tensor(norm_gen, dtype=torch.float32, device=args.device)
        Z_train = torch.tensor(norm_train, dtype=torch.float32, device=args.device)

        scores = []
        chunk_size = 500
        for idx in range(0, Z_g.shape[0], chunk_size):
            chunk_g = Z_g[idx:idx+chunk_size]
            dist_sq = torch.cdist(chunk_g, Z_train, p=2).pow(2)
            density_matrix = torch.exp(-dist_sq / (2 * args.sigma ** 2))
            chunk_scores = density_matrix.mean(dim=1).cpu().numpy()
            scores.extend(chunk_scores)

        scores = np.array(scores)

        # Attach scores to samples
        for idx, score in enumerate(scores):
            samples[idx]['score'] = float(score)

        # Sort samples by score descending (highest density first)
        sorted_samples = sorted(samples, key=lambda x: x['score'], reverse=True)

        # 4. Filter Top-6, Mid-6, Bottom-6
        # Top-6: Highest scores
        top_6 = sorted_samples[:6]
        
        # Mid-6: Around the median
        mid_start = max(0, len(sorted_samples) // 2 - 3)
        mid_6 = sorted_samples[mid_start:mid_start+6]
        
        # Bottom-6: Lowest scores
        bottom_6 = sorted_samples[-6:]

        print(f"  Top-6 LDS range:    {top_6[0]['score']:.4f} -> {top_6[-1]['score']:.4f}")
        print(f"  Mid-6 LDS range:    {mid_6[0]['score']:.4f} -> {mid_6[-1]['score']:.4f}")
        print(f"  Bottom-6 LDS range: {bottom_6[0]['score']:.4f} -> {bottom_6[-1]['score']:.4f}")

        # 5. Plot and save visual representations
        plot_trajectories(top_6, cat, "Top-6", output_dir, args.features)
        plot_trajectories(mid_6, cat, "Mid-6", output_dir, args.features)
        plot_trajectories(bottom_6, cat, "Bottom-6", output_dir, args.features)

        # 5b. Generate skeleton animation GIFs if requested (Benchpress only)
        if args.generate_gifs and args.dataset_name == 'benchpress':
            print(f"  Generating skeleton GIF animations for selected '{cat}' samples...")
            for group_name, group_data in [("Top-6", top_6), ("Mid-6", mid_6), ("Bottom-6", bottom_6)]:
                for s in group_data:
                    group_dir = os.path.join(output_dir, cat, group_name.lower().replace('-', '_'), s["sample_name"])
                    os.makedirs(group_dir, exist_ok=True)
                    
                    # Construct features config dict for the animator
                    features_dict = {feat: s["x_t"][f_idx].astype(float).tolist() for f_idx, feat in enumerate(args.features)}
                    
                    # Generate the three animator gifs
                    RearV_BenchpressAnimator(features_dict).animate(os.path.join(group_dir, "rear.gif"))
                    TopV_BenchpressAnimator(features_dict).animate(os.path.join(group_dir, "top.gif"))
                    LateralV_BenchpressAnimator(features_dict).animate(os.path.join(group_dir, "lateral.gif"))

        # Store text descriptors in dictionary (excluding raw x_t matrices for JSON encoding)
        output_results[cat] = {
            "top-6": [{"sample_name": s["sample_name"], "lds_score": s["score"]} for s in top_6],
            "mid-6": [{"sample_name": s["sample_name"], "lds_score": s["score"]} for s in mid_6],
            "bottom-6": [{"sample_name": s["sample_name"], "lds_score": s["score"]} for s in bottom_6]
        }

    # 6. Save selection metadata to JSON
    json_path = os.path.join(output_dir, "lds_selected_samples.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=4)

    print(f"\nLDS selection complete!")
    print(f"JSON Metadata saved: {json_path}")
    print(f"Visualizations saved to: {output_dir}")

    # Generate a formatted markdown summary report
    print("\n--- LDS Selection Summary Report ---")
    print(f"| Category | Group | Sample Name | LDS Score |")
    print(f"| :--- | :--- | :--- | :--- |")
    for cat in output_results:
        for group in ["top-6", "mid-6", "bottom-6"]:
            for item in output_results[cat][group]:
                print(f"| {cat} | {group.upper()} | {item['sample_name']} | {item['lds_score']:.5f} |")


if __name__ == '__main__':
    main()
