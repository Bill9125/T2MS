import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import yaml

def plot_loss_curve(loss_list, save_path, filename='loss_curve.png', val_loss_list=None):
    if len(loss_list) == 0:
        print("loss_list is empty, skipping plotting.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Training Loss', color='blue', alpha=0.8)
    if val_loss_list is not None and len(val_loss_list) > 0:
        if len(val_loss_list) == len(loss_list):
            plt.plot(val_loss_list, label='Validation Loss', color='orange', alpha=0.8)
        else:
            val_x = np.linspace(0, len(loss_list) - 1, len(val_loss_list))
            plt.plot(val_x, val_loss_list, label='Validation Loss', color='orange', alpha=0.8)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
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
    args.config = os.path.join('config', f'{args.dataset_name}.yaml')
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        args.dataset_root = config.get('dataset_root', './Data')
        args.general_seed = config.get('general_seed', 2025)
        
        args.features = [feature[0]["name"] for feature in config["features"].values()]
        args.flow_dim = config.get('flow_dim', 128)
        args.input_dim = config.get('input_dim', 10)
        args.split_base_num = config['dataset'].get('split_base_num', 36)
        # yaml 中的 caption 是訓練專用的
        args.train_caption = config['dataset'].get('caption', 'explain')
        
        # 若從 argparse 已經傳入 caption (推論用)，就不覆蓋；否則給預設值
        if not getattr(args, 'caption', None):
            args.caption = 'explain'

        args.embedding_dim = config['vae'].get('embedding_dim', 64)
        args.block_hidden_size = config['vae'].get('block_hidden_size', 128)
        args.num_residual_layers = config['vae'].get('num_residual_layers', 3)
        args.res_hidden_size = config['vae'].get('res_hidden_size', 256)
        args.pretrained_epc = config['vae'].get('epoch', 80000)
        
        args.denoiser = config['diffusion'].get('denoiser', 'DiT')
        args.backbone = config['diffusion'].get('backbone', 'flowmatching')
        
        # CLIP config
        clip_cfg = config.get('clip', {})
        args.clip_dim = int(clip_cfg.get('clip_dim', 256))
        args.clip_temperature = float(clip_cfg.get('temperature', 0.07))
        args.text_backbone = clip_cfg.get('text_backbone', 'all-MiniLM-L6-v2')
        args.clip_epoch = int(clip_cfg.get('epoch', 500))
        args.clip_lr = float(clip_cfg.get('learning_rate', 1e-4))
        args.clip_batch_size = int(clip_cfg.get('batch_size', 128))
    return args
