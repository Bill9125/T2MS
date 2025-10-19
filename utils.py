import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import yaml

def plot_loss_curve(loss_list, save_path, filename='loss_curve.png'):
    if len(loss_list) == 0:
        print("loss_list is empty, skipping plotting.")
        return
    plt.figure(figsize=(10, 6))
    x = [i for i in range(len(loss_list))]
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"Loss curve saved to {full_path}")
    
def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        
def get_cfg(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}: {config}")
        args.dataset_root = config.get('dataset_root', './Data')
        args.general_seed = config.get('general_seed', 2025)
        
        cfg = config[args.dataset_name]
        args.flow_dim = cfg.get('flow_dim', 128)
        
        args.split_base_num = cfg['dataset'].get('split_base_num', 36)
        args.caption = cfg['dataset'].get('caption', 'Caption_explain_no_barbell')

        args.embedding_dim = cfg['vae'].get('embedding_dim', 64)
        args.block_hidden_size = cfg['vae'].get('block_hidden_size', 128)
        args.num_residual_layers = cfg['vae'].get('num_residual_layers', 3)
        args.res_hidden_size = cfg['vae'].get('res_hidden_size', 256)
        
        args.denoiser = cfg['diffusion'].get('denoiser', 'DiT')
        args.backbone = cfg['diffusion'].get('backbone', 'flowmatching')
    return args