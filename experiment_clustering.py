import os
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

def load_vae_model(args):
    print('Loading pretrained VAE encoder from: ', args.pretrainedvae_path)
    vae = vqvae(args).to(args.device).float().eval()
    vae.load_state_dict(torch.load(args.pretrainedvae_path, map_location=args.device))
    return vae.encoder

def main():
    parser = argparse.ArgumentParser(description="VAE Embedding PCA and t-SNE Clustering Experiment")
    parser.add_argument('--dataset_name', '-d', type=str, default='benchpress', help='dataset name (benchpress or deadlift)')
    parser.add_argument('--subject_mode', type=str, default='mix', choices=['isolated', 'mix'], help='subject type')
    parser.add_argument('--save_path', type=str, default='./results/clustering_experiment', help='Save path')
    parser.add_argument('--all_subjects', type=str, default='true', help='Use all subjects in dataset instead of a 20-subject subset (true/false)')
    parser.add_argument('--perplexity', type=int, default=None, help='t-SNE perplexity (default: auto-adjust based on data size)')
    args = parser.parse_args()

    # Parse boolean flags
    args.all_subjects = args.all_subjects.lower() == 'true'

    # Load configuration
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"All subjects mode: {args.all_subjects}")

    # Set up VAE path
    args.pretrainedvae_path = os.path.join(
        './results/saved_pretrained_models', 
        f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject_mode}', 
        'final_model.pth'
    )
    if not os.path.exists(args.pretrainedvae_path):
        # Fallback to the other mode if the selected one doesn't exist
        other_mode = "isolated" if args.subject_mode == "mix" else "mix"
        alt_path = os.path.join(
            './results/saved_pretrained_models', 
            f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{other_mode}', 
            'final_model.pth'
        )
        if os.path.exists(alt_path):
            print(f"VAE model not found at {args.pretrainedvae_path}. Falling back to {alt_path}")
            args.pretrainedvae_path = alt_path
        else:
            raise FileNotFoundError(f"No VAE model found at {args.pretrainedvae_path} or {alt_path}")

    # Load VAE Encoder
    vae_encoder = load_vae_model(args)

    # Determine movement categories based on dataset
    if args.dataset_name == 'benchpress':
        categories = ['correct', 'tilting_to_the_left', 'tilting_to_the_right', 'scapular_protraction', 'elbows_flaring']
    elif args.dataset_name == 'deadlift':
        categories = ['Correct', 'Barbell_moving_away_from_the_shins', 'Barbell_colliding_with_the_knees', 'Lower_back_rounding', 'Hips_rising_before_the_barbell_leaves_the_ground']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Load dataset index
    json_path = os.path.join(args.dataset_root, args.dataset_name, 'data.json')
    print(f"Loading data index from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # Deterministic seeds
    random.seed(args.general_seed)
    np.random.seed(args.general_seed)

    # 1. Subject Selection
    if args.all_subjects:
        chosen_subjects = sorted(list(all_data.keys()))
        print(f"\nLoaded all subjects from the dataset. Total: {len(chosen_subjects)}")
    else:
        # Select 4 subjects per category to get exactly 20 subjects
        category_to_subjects = {cat: [] for cat in categories}
        for subj in all_data.keys():
            matched_cats = [cat for cat in categories if cat in subj]
            if len(matched_cats) == 1:
                category_to_subjects[matched_cats[0]].append(subj)
            elif len(matched_cats) > 1:
                category_to_subjects[matched_cats[0]].append(subj)

        chosen_subjects = []
        print("\nSubject selection summary (4 per category):")
        for cat in categories:
            subjs = sorted(category_to_subjects[cat])
            if len(subjs) >= 4:
                selected = random.sample(subjs, 4)
            else:
                selected = subjs
                print(f"Warning: Category '{cat}' only has {len(subjs)} subjects. Taking all.")
            chosen_subjects.extend(selected)
            print(f" - {cat}: {len(selected)} subjects selected out of {len(subjs)}")

        print(f"\nTotal selected subjects: {len(chosen_subjects)}")
        print("Selected subjects list:")
        for i, s in enumerate(chosen_subjects, 1):
            print(f" {i:02d}. {s}")

    # 2. Extract and pad all clips from selected subjects
    data_list = []
    labels_class = []
    labels_subject = []

    # Same target length interpolation as in dataset.py test mode
    target_len = args.split_base_num * 2

    for subj in chosen_subjects:
        # Identify main class for this subject
        main_class = next((cat for cat in categories if cat in subj), "unknown")
        if main_class == "unknown":
            continue
        
        clips = all_data[subj]
        for clip_id, feat_dict in clips.items():
            # Gather all features
            seqs_T = []
            T_list = []
            for k in args.features:
                if k not in feat_dict:
                    continue
                x = torch.as_tensor(feat_dict[k], dtype=torch.float32)
                if x.dim() == 1:
                    seqs_T.append(x)
                    T_list.append(x.size(0))
                else:
                    raise ValueError(f"Feature '{k}' must be [T], got {tuple(x.size())}")

            # Consistency check
            if len(seqs_T) != len(args.features) or len(set(T_list)) != 1:
                continue

            x_nfT = torch.stack(seqs_T, dim=0) # [n_f, T]
            
            # Align length to target_len (adaptive pooling or interpolation)
            Tcur = x_nfT.size(1)
            Ttar = target_len
            x_1cT = x_nfT.unsqueeze(0) # [1, n_f, T]
            if Tcur > Ttar:
                x_1cT = F.adaptive_avg_pool1d(x_1cT, output_size=Ttar)
            elif Tcur < Ttar:
                x_1cT = F.interpolate(x_1cT, size=Ttar, mode='linear', align_corners=True)
            x_nfT = x_1cT.squeeze(0)

            data_list.append(x_nfT)
            labels_class.append(main_class)
            # Extract subject base ID (e.g. subject_101 -> 101)
            subj_parts = subj.split('_')
            subj_id = subj_parts[1] if len(subj_parts) > 1 else subj
            labels_subject.append(subj_id)

    if not data_list:
        print("Error: No valid clips found for the selected subjects.")
        return

    print(f"\nSuccessfully loaded {len(data_list)} clips from {len(chosen_subjects)} subjects.")

    # Stack into a batch tensor
    xs_tensor = torch.stack(data_list).to(args.device) # [B, n_f, T]

    # 3. Pass through VAE encoder
    print("Extracting VAE embeddings...")
    with torch.no_grad():
        z, _ = vae_encoder(xs_tensor) # [B, embedding_dim, flow_dim]

    # Two types of embeddings: Mean-pooled and Flattened
    embs_mean = z.mean(dim=-1).cpu().numpy() # [B, embedding_dim]
    embs_flat = z.flatten(start_dim=1).cpu().numpy() # [B, embedding_dim * flow_dim]

    print(f"Mean-pooled embeddings shape: {embs_mean.shape}")
    print(f"Flattened embeddings shape: {embs_flat.shape}")

    # Prepare save folder
    os.makedirs(args.save_path, exist_ok=True)

    # 4. Perform Clustering, Dimensionality Reduction & Visualization
    for emb_name, embs in [("Mean-pooled", embs_mean), ("Flattened", embs_flat)]:
        print(f"\n--- Analysis for {emb_name} Embeddings ({args.dataset_name}) ---")

        unique_subjects = sorted(list(set(labels_subject)))
        unique_classes = sorted(list(set(labels_class)))

        # Compute silhouette scores to quantitatively evaluate if they group
        class_silhouette = silhouette_score(embs, labels_class)
        # Compute subject silhouette only if there are multiple subjects and clips
        if len(unique_subjects) > 1 and len(labels_subject) > len(unique_subjects):
            subject_silhouette = silhouette_score(embs, labels_subject)
        else:
            subject_silhouette = 0.0

        print(f"Silhouette Score (by Movement Class): {class_silhouette:.4f}")
        print(f"Silhouette Score (by Subject ID): {subject_silhouette:.4f}")

        # Core dimensionality reduction code matches evaluate/visualization.py:L146-L164 exactly
        # PCA
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(embs)

        # t-SNE
        n = embs.shape[0]
        if args.perplexity is not None:
            perplexity = max(2, min(n - 1, args.perplexity))
        else:
            # Auto-adjust: use 100 for large datasets, 30 for small datasets
            if n > 1000:
                perplexity = 100
            else:
                perplexity = max(2, min(n - 1, 30))
        tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=args.general_seed)
        combined_tsne = tsne.fit_transform(embs)

        # HUE Legend controls: Hiding subject legend if there are too many subjects to prevent UI clutter
        show_subject_legend = len(unique_subjects) <= 20

        # Plot 1: Colored by Movement Class
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        class_palette = sns.color_palette("Set1", len(unique_classes))
        sns.scatterplot(x=combined_pca[:, 0], y=combined_pca[:, 1], hue=labels_class, ax=axs[0], alpha=0.8, palette=class_palette, s=40)
        axs[0].set_title(f'PCA - Colored by Movement Class\n(Silhouette: {class_silhouette:.4f})', fontsize=14)
        axs[0].legend(frameon=True, shadow=True, title="Movement Class")
        axs[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(x=combined_tsne[:, 0], y=combined_tsne[:, 1], hue=labels_class, ax=axs[1], alpha=0.8, palette=class_palette, s=40)
        axs[1].set_title(f't-SNE - Colored by Movement Class\n(Perplexity: {perplexity})', fontsize=14)
        axs[1].legend(frameon=True, shadow=True, title="Movement Class")
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f'{emb_name} Embeddings ({len(unique_subjects)} Subjects, {len(labels_class)} clips) - {args.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_save_path_class = os.path.join(args.save_path, f"clustering_by_class_{emb_name.lower().replace('-', '_')}_{args.dataset_name}.png")
        plt.savefig(plot_save_path_class, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_save_path_class}")

        # Plot 2: Colored by Subject ID
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        
        # Color palette for Subject IDs
        if len(unique_subjects) <= 20:
            subject_palette = sns.color_palette("tab20", len(unique_subjects))
        else:
            subject_palette = sns.color_palette("husl", len(unique_subjects))

        sns.scatterplot(x=combined_pca[:, 0], y=combined_pca[:, 1], hue=labels_subject, ax=axs[0], alpha=0.8, palette=subject_palette, s=40, legend='brief' if show_subject_legend else False)
        axs[0].set_title(f'PCA - Colored by Subject ID\n(Silhouette: {subject_silhouette:.4f})', fontsize=14)
        if show_subject_legend:
            axs[0].legend(frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left', title="Subject ID")
        axs[0].grid(True, linestyle='--', alpha=0.5)

        sns.scatterplot(x=combined_tsne[:, 0], y=combined_tsne[:, 1], hue=labels_subject, ax=axs[1], alpha=0.8, palette=subject_palette, s=40, legend='brief' if show_subject_legend else False)
        axs[1].set_title(f't-SNE - Colored by Subject ID\n(Perplexity: {perplexity})', fontsize=14)
        if show_subject_legend:
            axs[1].legend(frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left', title="Subject ID")
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f'{emb_name} Embeddings ({len(unique_subjects)} Subjects, {len(labels_class)} clips) - {args.dataset_name.capitalize()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_save_path_subject = os.path.join(args.save_path, f"clustering_by_subject_{emb_name.lower().replace('-', '_')}_{args.dataset_name}.png")
        plt.savefig(plot_save_path_subject, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_save_path_subject}")

    print(f"\nClustering Experiment for '{args.dataset_name}' Completed successfully!")
    print(f"All plots have been saved to the directory: {os.path.abspath(args.save_path)}")

if __name__ == '__main__':
    main()
