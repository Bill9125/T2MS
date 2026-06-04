import argparse
import torch
from matplotlib import pyplot as plt
from model.denoiser.mlp import MLP
from model.denoiser.mymlp import myMLP
from model.denoiser.mytransformer import Transformer
from model.pretrained.myvqvae import vqvae
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
import os
import numpy as np
import math
from pretrained_mylavae import plot_pca_tsne
from tqdm import tqdm
from utils import *
# pyrefly: ignore [missing-import]
from scipy.signal import savgol_filter

def plot_side_by_side_comparison(args, x_1, x_t, subjects_list):
    save_path = args.generation_save_path_result
    for i in range(len(x_1)):
        fig_path = os.path.join(save_path, f'sample_{i}.jpg')
        plt.clf()
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{subjects_list[i]}', fontsize=10)

        # 左圖：ground truth
        ax1 = plt.subplot(1, 2, 1)
        for j in range(len(x_1[i])):
            ax1.plot(x_1[i][j], label=f"{args.features[j]}")
        ax1.set_title('Ground Truth')

        # 右圖：generated
        ax2 = plt.subplot(1, 2, 2)
        for j in range(len(x_t[i])):
            ax2.plot(x_t[i][j], label=f"{args.features[j]}")
        ax2.set_title('Generated')

        plt.savefig(fig_path)
        plt.close()
        
def save_result(root, features, visualization=True):
    # save predict sample
    if visualization:
        print(features.keys())
        rear = os.path.join(root, f'rear.gif')
        top = os.path.join(root, f'top.gif')
        lateral = os.path.join(root, f'lateral.gif')
        RearV_BenchpressAnimator(features).animate(rear)
        TopV_BenchpressAnimator(features).animate(top)
        LateralV_BenchpressAnimator(features).animate(lateral)

def infer(args, run_id):
    step = args.total_step
    cfg_scale = args.cfg_scale
    device = args.device

    print(f"Inference config::Step: {step}\t CFG Scale: {cfg_scale}\t use_clip: {args.use_clip}")
    if args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    elif args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
    
    _, test_loader = loader_provider(args, period='test')
    print('dataset length:', len(test_loader))
    vae = vqvae(args).to(device).float().eval()
    state = torch.load(args.pretrainedvae_path, map_location=device, weights_only=False)
    vae.load_state_dict(state)
    pretrained_model = vae
    model = {'DiT': Transformer(args.flow_dim), 'MLP': MLP(), 'myMLP': myMLP(in_channels=args.embedding_dim, cond_dim=args.flow_dim, seq_len=args.flow_dim)}.get(args.denoiser)
    if model:
        model = model.to(args.device)
    else:
        raise ValueError(f"No denoiser found")

    # --- CLIP Text Encoder (if using CLIP mode) ---
    text_encoder = None
    if args.use_clip:
        clip_ckpt = torch.load(args.clip_model_path, map_location=device, weights_only=False)
        clip_dim = clip_ckpt['clip_dim']
        text_emb_dim = clip_ckpt.get('text_emb_dim', 128)
        
        # Check if text_encoder is present in checkpoint (backward compatibility)
        if 'text_encoder' in clip_ckpt:
            from model.pretrained.text_encoder import TextEncoder
            text_encoder = TextEncoder(input_dim=text_emb_dim, clip_dim=clip_dim).to(device)
            text_encoder.load_state_dict(clip_ckpt['text_encoder'])
            for param in text_encoder.parameters():
                param.requires_grad = False
            text_encoder.eval()
            print(f"  CLIP Text Encoder loaded (input_dim={text_emb_dim}, clip_dim={clip_dim})")
        else:
            print(f"  CLIP checkpoint does not contain text_encoder. Bypassing and using raw {clip_dim}-dim embeddings directly.")

        # Replace text_proj in denoiser to match clip_dim
        if hasattr(model, 'text_proj'):
            embed_dim = model.embed_dim
            model.text_proj = torch.nn.Linear(clip_dim, embed_dim).to(device)

    model.encoder = pretrained_model.encoder
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device, weights_only=False)['model'])
    model.to(device).eval()
    backbone = {'flowmatching': RectifiedFlow(), 'ddpm': DDPM(args.total_step, args.device)}.get(args.backbone)
    if backbone:
        if args.backbone == 'flowmatching':
            rf = backbone
        elif args.backbone == 'ddpm':
            ddpm = backbone
    else:
        raise ValueError(f"No backbone found")

    x_1_list = []
    x_t_list = []
    y_list = []
    frames_list = []
    x_infer_list = []
    subjects_list = []
    with (torch.no_grad()):
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Generating Batches")):
            # 支援 custom_collate_fn 回傳的 list 格式
            if not isinstance(batch_data, list):
                batch_data = [batch_data]
            
            for data in batch_data:
                features = {feat : {} for feat in args.features[-args.input_dim:]}
                y, x_1, embedding, subject, clip = data
                y_list.append(y)
                
                x_1 = x_1.float().to(device)

                # --- Text conditioning ---
                if args.use_clip:
                    embedding = embedding.float().to(device)
                    if text_encoder is not None:
                        # Backward compatibility
                        text_cond = text_encoder(embedding)   # [B, clip_dim]
                    else:
                        text_cond = embedding  # [B, clip_dim]
                    embedding = model.text_proj(text_cond) # [B, embed_dim]
                else:
                    embedding = embedding.float().to(device)

                x_t, before = model.encoder(x_1)
                x_t_latent_enc = x_t.clone()
                
                # 若啟用 fixed_noise，則限制在 10 個固定的「雜訊區間(Seed)」
                if getattr(args, 'fixed_noise', True):
                    x_t = torch.empty(x_t.shape, device=device)
                    torch.nn.init.trunc_normal_(
                        x_t,
                        mean=0.0, 
                        std=1.0, 
                        a=2.0,
                        b=3.0
                    )
                else:
                    x_t = torch.randn_like(x_t).float().to(device)
                
                # 針對擴散步數加上 tqdm
                for j in tqdm(range(step), desc=f"Inference Steps (Batch {batch_idx})", leave=False):
                    if args.backbone == 'flowmatching':
                        t = torch.round(torch.full((x_t.shape[0],), j * 1.0 / step, device=device) * step) / step
                        pred_uncond = model(input=x_t, t=t, text_input=None)
                        pred_cond = model(input=x_t, t=t, text_input=embedding)
                        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                        x_t = rf.euler(x_t, pred, 1.0 / step)
                        
                    elif args.backbone == 'ddpm':
                        t = torch.full((x_t.size(0),), math.floor(step-1-j), dtype=torch.long, device=device)
                        pred_uncond = model(input=x_t, t=t, text_input=None)
                        pred_cond = model(input=x_t, t=t, text_input=embedding)
                        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                        x_t = ddpm.p_sample(x_t, pred, t)

                    if batch_idx == 0:
                        x_t_infer_stat, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
                        x_t_infer_stat = x_t_infer_stat.detach().cpu().numpy().squeeze()
                        x_infer_list.append(x_t_infer_stat[0])
                        
                    if (j % 100 == 0) or (j == step - 1):
                        xt_decode, _ = pretrained_model.decoder(x_t, length=x_1.shape[-1])
                        xt_decode_np = xt_decode.detach().cpu().numpy().squeeze()
                        frames_list.append(xt_decode_np.copy())
                
                x_t, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
                if batch_idx == 0:
                    x_t_infer_gt, after = pretrained_model.decoder(x_t_latent_enc, length=x_1.shape[-1])
                    x_t_infer_gt = x_t_infer_gt.detach().cpu().numpy().squeeze()
                    x_infer_list.append(x_t_infer_gt[0])

                x_1_np = x_1.detach().cpu().numpy().squeeze()
                x_t_np = x_t.detach().cpu().numpy().squeeze()
                
                # 加上 Savitzky-Golay 濾波器
                try:
                    x_t_np = savgol_filter(x_t_np, window_length=7, polyorder=2, axis=-1)
                except Exception as e:
                    pass
                
                x_1_list.append(x_1_np)
                x_t_list.append(x_t_np)
                subjects_list.append(f"{subject[0]}_{clip[0]}")
                
                for i, key in enumerate(features.keys()):
                    features[key] = x_t_np[i].astype(float).tolist()
                
                save_path = os.path.join(args.generation_save_path_result, f'{subject[0]}_{clip[0]}')
                os.makedirs(save_path, exist_ok=True)
                if args.visualization:
                    save_result(save_path, features)
                np.save(os.path.join(save_path, f'x_t.npy'), x_t_np)
                np.save(os.path.join(save_path, f'x_1.npy'), x_1_np)
                np.save(os.path.join(save_path, f'embedding.npy'), embedding.detach().cpu().numpy())
    
    if args.visualization:
        plot_side_by_side_comparison(args, x_1_list, x_t_list,  subjects_list)
        plot_pca_tsne(x_1_list, x_t_list, args.generation_save_path_result)
    return x_1_list, x_t_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference flow matching model")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
    
    parser.add_argument('--cfg_scale', type=int, default=3, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='total step sampled from [0,1]')
    parser.add_argument('--subject', type=str, choices=['isolated','mix'], help='subject type')
    parser.add_argument('--caption', type=str, choices=['explain','style_new'], help='caption type for inference')

    # for inference
    parser.add_argument('--checkpoint_id', type=int, default=1000,help='model id')
    parser.add_argument('--dataset_name', '-d', type=str, choices=['deadlift', 'benchpress'], help='dataset name')
    parser.add_argument('--denoiser', type=str, default='DiT', help='denoiser type [DiT, MLP, myMLP]')
    parser.add_argument('--backbone', type=str, default='flowmatching', help='backbone type [flowmatching, ddpm]')
    parser.add_argument('--visualization', action='store_true', help='visualization')
    parser.add_argument('--fixed_noise', action='store_true', help='Evaluate fixed random noise dataset')
    parser.add_argument('--run_time', type=int, default=1, help='Number of runs')

    # CLIP-specific arguments
    parser.add_argument('--use_clip', action='store_true',
                        help='Use CLIP-aligned text encoder instead of pre-stored embeddings')
    parser.add_argument('--clip_model_path', type=str, default=None,
                        help='Path to Stage 1 CLIP model checkpoint')
    parser.add_argument('--pretrainedvae_path', type=str, default=None,
                        help='Path to pretrained VAE (Stage 2 checkpoint)')

    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name +'.yaml')
    args = get_cfg(args)
    if not args.pretrainedvae_path:
        vae_dir = f'clip_{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}' if args.use_clip else f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}'
        args.pretrainedvae_path = os.path.join('./results/saved_pretrained_models', vae_dir, 'final_model.pth')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clip_tag = 'clip' if args.use_clip else 'legacy'
    args.checkpoint_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}_{}_{}_{}'.format(
        args.backbone, args.denoiser, args.dataset_name, args.train_caption, args.pretrained_epc, args.subject, clip_tag), 'model_{}.pth'.format(args.checkpoint_id))
    
    for i in range(args.run_time):
        print(f'--- Run {i+1}/{args.run_time} ---')
        args.generation_save_path_result = os.path.join(args.save_path, f'generation_{"fixed" if args.fixed_noise else "random"}', f'{args.backbone}_{args.denoiser}_{args.dataset_name}_{args.cfg_scale}_{args.total_step}_{args.subject}_{args.caption}_{clip_tag}', f'run_{i}')
        os.makedirs(args.generation_save_path_result, exist_ok=True)
        x_1_list, x_t_list = infer(args, run_id=i)
