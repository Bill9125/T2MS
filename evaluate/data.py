import os
import glob
import numpy as np
import torch
from model.pretrained.myvqvae import vqvae

def load_vae_model(args):
    print('Loading pretrained VAE encoder from: ', args.pretrainedvae_path)
    vae = vqvae(args).to(args.device).float().eval()
    vae.load_state_dict(torch.load(args.pretrainedvae_path, map_location=args.device))
    return vae.encoder

def load_training_data(args, vae_encoder):
    if 'NND' not in args.method_list and 'TSNE' not in args.method_list and 'LDS' not in args.method_list:
        return None, None, None, None

    if args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
    elif args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    
    train_loader, _ = loader_provider(args, period='train')
    print(f"Loading training data for Novelty/t-SNE... (Total batches: {len(train_loader)})")
    
    train_repr_list, train_repr_fid_list, train_labels_list, train_embs_list = [], [], [], []
    
    with torch.no_grad():
        for batch_data in train_loader:
            if isinstance(batch_data, list):
                texts, xs, embs, subs, clips = batch_data[0]
            else:   
                texts, xs, embs, subs, clips = batch_data
            
            xs = xs.float().to(args.device)
            features, _ = vae_encoder(xs)
            
            train_repr_list.append(features.flatten(start_dim=1).cpu().numpy())
            train_repr_fid_list.append(features.mean(dim=-1).cpu().numpy())
            train_embs_list.append(embs.cpu().numpy())
            
            for s in subs:
                s_str = s[0] if isinstance(s, tuple) else s
                train_labels_list.append(str(s_str))
    
    train_repr_my = np.concatenate(train_repr_list, axis=0)
    train_repr_fid_my = np.concatenate(train_repr_fid_list, axis=0)
    train_embs_my = np.concatenate(train_embs_list, axis=0)
    train_labels_my = np.array(train_labels_list)
    
    print(f"Loaded {len(train_labels_my)} training labels. Sample: {train_labels_my[:10]}")
    return train_repr_my, train_repr_fid_my, train_embs_my, train_labels_my

def load_generated_data(args):
    x_1_list, x_t_list, emb_list = [], [], []
    x_t_labels = []
    grouped_samples = {}
    if args.dataset_name == 'benchpress':
        known_classes = ['correct', 'tilting_to_the_left', 'tilting_to_the_right', 'scapular_protraction', 'elbows_flaring']
    elif args.dataset_name == 'deadlift':
        known_classes = ['Correct', 'Barbell_moving_away_from_the_shins', 'Barbell_colliding_with_the_knees', 'Lower_back_rounding', 'Hips_rising_before_the_barbell_leaves_the_ground']
    else:
        known_classes = []

    run_folders = glob.glob(os.path.join(args.generation_save_path, "run_*"))
    if not run_folders:
        print(f"No run folders found in {args.generation_save_path}")
        return None, None, None, None

    print(f"Found {len(run_folders)} run folders for evaluation.")
    for run_save_path in sorted(run_folders, key=lambda x: int(os.path.basename(x).split('_')[-1])):
        for sample_dir in os.listdir(run_save_path):
            sample_path = os.path.join(run_save_path, sample_dir)
            if not os.path.isdir(sample_path) or sample_dir.startswith('.'):
                continue
                
            x_t_path = os.path.join(sample_path, 'x_t.npy')
            x_1_path = os.path.join(sample_path, 'x_1.npy')
            emb_path = os.path.join(sample_path, 'embedding.npy')
            
            if os.path.exists(x_t_path) and os.path.exists(x_1_path):
                error_class = next((k for k in known_classes if k in sample_dir), "unknown")
                if error_class == "unknown":
                    continue
                    
                x_t = np.load(x_t_path)
                x_1 = np.load(x_1_path)
                
                if os.path.exists(emb_path):
                    emb = np.load(emb_path)
                    emb = np.expand_dims(emb, axis=0) if emb.ndim == 1 else emb
                    emb = emb.squeeze(1) if emb.ndim == 3 else emb
                else:
                    emb = np.zeros((1, args.embedding_dim))
                
                x_t_list.append(x_t)
                x_1_list.append(x_1)
                emb_list.append(emb)
                x_t_labels.append(error_class)
                
                if error_class not in grouped_samples:
                    grouped_samples[error_class] = {'x_1': [], 'x_t': [], 'emb': []}
                grouped_samples[error_class]['x_1'].append(x_1)
                grouped_samples[error_class]['x_t'].append(x_t)
                grouped_samples[error_class]['emb'].append(emb)
                    
    return x_1_list, x_t_list, emb_list, grouped_samples, x_t_labels
