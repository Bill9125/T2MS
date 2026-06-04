"""
Stage 3: Diffusion Training with CLIP-aligned Text Encoder + LA Encoder

Trains the diffusion denoiser (DiT / MLP) using:
  - Frozen CLIP-aligned LA Encoder (from Stage 1)
  - Frozen CLIP-aligned Text Encoder (from Stage 1)
  - Frozen LA Decoder (from Stage 2)

The Text Encoder encodes captions on-the-fly to produce text conditioning,
replacing the pre-stored embedding approach.

Supports two modes via --use_clip flag:
  --use_clip: Uses the new CLIP pipeline (requires --clip_model_path)
  (default): Falls back to the original pre-stored embedding pipeline

Usage:
    # CLIP mode
    python mytrain.py -d benchpress --subject mix --use_clip \\
        --clip_model_path ./results/clip_models/benchpress_mix_dim256/final_clip_model.pth \\
        --pretrained_model_path ./results/saved_pretrained_models/clip_.../final_model.pth

    # Legacy mode (unchanged behavior)
    python mytrain.py -d benchpress --subject mix
"""

import argparse
import os
import torch
from torch.optim import AdamW, lr_scheduler
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from model.denoiser.mytransformer import Transformer
from model.denoiser.mlp import MLP
from model.denoiser.mymlp import myMLP
from model.pretrained.myvqvae import vqvae
from tqdm import tqdm
from utils import get_cfg, plot_loss_curve, seed_everything
import numpy as np

def train(args):
    print(f"Training config::\tepoch: {args.epochs}\tsave_path: {args.save_path}\tdevice: {args.device}")
    print(f"  use_clip: {args.use_clip}")
    os.makedirs(args.save_path, exist_ok=True)

    if args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    elif args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
    train_loader, test_loader = loader_provider(args)

    # --- CLIP Text Encoder (if using CLIP mode) ---
    text_encoder = None
    if args.use_clip:
        clip_ckpt = torch.load(args.clip_model_path, map_location=args.device, weights_only=False)
        clip_dim = clip_ckpt['clip_dim']
        text_emb_dim = clip_ckpt.get('text_emb_dim', 128)

        # Check if text_encoder is in checkpoint (backward compatibility)
        if 'text_encoder' in clip_ckpt:
            from model.pretrained.text_encoder import TextEncoder
            text_encoder = TextEncoder(input_dim=text_emb_dim, clip_dim=clip_dim).to(args.device)
            text_encoder.load_state_dict(clip_ckpt['text_encoder'])
            # Freeze text encoder
            for param in text_encoder.parameters():
                param.requires_grad = False
            text_encoder.eval()
            print(f"  CLIP Text Encoder loaded (input_dim={text_emb_dim}, clip_dim={clip_dim})")
        else:
            print(f"  CLIP checkpoint does not contain text_encoder. Bypassing and using raw {clip_dim}-dim embeddings directly.")

        # Use clip_dim as the conditioning dimension for the denoiser
        cond_dim = clip_dim
    else:
        # Legacy: conditioning dim = flow_dim (pre-stored embedding size matches this via text_proj)
        cond_dim = args.flow_dim

    # --- Denoiser ---
    model = {
        'DiT': Transformer(args.flow_dim, embedding_dim=args.embedding_dim),
        'MLP': MLP(),
        'myMLP': myMLP(in_channels=args.embedding_dim, cond_dim=args.flow_dim, seq_len=args.flow_dim),
    }.get(args.denoiser)
    if model:
        model = model.to(args.device)
    else:
        raise ValueError(f"No denoiser found")

    # If CLIP mode, replace the text_proj to match clip_dim → embed_dim (128)
    if args.use_clip and hasattr(model, 'text_proj'):
        embed_dim = model.embed_dim  # 128 for DiT
        model.text_proj = torch.nn.Linear(cond_dim, embed_dim).to(args.device)
        print(f"  Replaced text_proj: Linear({cond_dim} → {embed_dim})")

    # --- Pretrained VAE (encoder + decoder) ---
    pretrained_model = vqvae(args).float().to(args.device)
    pretrained_model.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device(args.device), weights_only=False))

    # --- Backbone ---
    backbone = {'flowmatching': RectifiedFlow(), 'ddpm': DDPM(args.total_step, args.device)}.get(args.backbone)
    if not backbone:
        raise ValueError(f"No backbone found")

    # Attach frozen encoder to denoiser
    model.encoder = pretrained_model.encoder
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = not args.usepretrainedvae
    print(f"Total learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"VAE learnable parameters: {sum(p.numel() for p in pretrained_model.encoder.parameters() if p.requires_grad)}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=len(train_loader) * args.epochs)
    loss_list = []
    start_epoch = 0
    
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        loss_list = checkpoint['loss_list']
        plot_loss_curve(loss_list, args.save_path)

    print("training...")
    epoch_losses = []
    for epoch in range(start_epoch, args.epochs):
        group_losses = []
        for group in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            for (y_text, x_1, y_text_embedding, subject, clip) in group:
                x_1 = x_1.float().to(args.device)
                x_1, before = model.encoder(x_1)  # TS data ==>VAE==> clear TS embedding

                # --- Text conditioning ---
                if args.use_clip:
                    # Encode/use pre-stored text embedding
                    y_text_embedding = y_text_embedding.float().to(args.device)
                    if text_encoder is not None:
                        with torch.no_grad():
                            text_cond = text_encoder(y_text_embedding)  # [B, clip_dim]
                    else:
                        text_cond = y_text_embedding  # [B, clip_dim]
                    # Project to denoiser conditioning space
                    text_cond = model.text_proj(text_cond)  # [B, embed_dim]
                else:
                    # Legacy: use pre-stored embeddings directly
                    text_cond = y_text_embedding.float().to(args.device)

                if args.backbone == 'flowmatching':
                    t = torch.round(torch.rand(x_1.size(0)).to(args.device) * args.total_step) / args.total_step
                    x_t, x_0 = backbone.create_flow(x_1, t)  # x_t: dirty TS embedding, x_0：pure noise
                    noise_gt = x_1 - x_0
                elif args.backbone == 'ddpm':
                    t = torch.floor(torch.rand(x_1.size(0)).to(args.device) * args.total_step).long()
                    noise_gt = torch.randn_like(x_1).float().to(args.device)
                    x_t, n_xt = backbone.q_sample(x_1, t, noise_gt)
                else:
                    raise ValueError(f"Unsupported backbone type: {args.backbone}")

                optimizer.zero_grad()
                decide = torch.rand(1) < 0.1
                if decide:
                    text_cond = None
                pred = model(input=x_t, t=t, text_input=text_cond)
                loss = backbone.loss(pred, noise_gt)
                loss.backward()
                group_losses.append(loss.item())
                optimizer.step()
            scheduler.step()
        epoch_losses.append(np.mean(group_losses))
        print(f'[Epoch {epoch}] loss: {np.mean(group_losses):.5f}')

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(f'Saving model {epoch} to {args.save_path}...')
            plot_loss_curve(epoch_losses, args.save_path)
            save_dict = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                loss_list=loss_list,
                use_clip=args.use_clip,
            )
            if args.use_clip:
                save_dict['clip_model_path'] = args.clip_model_path
            torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))


def get_args():
    parser = argparse.ArgumentParser(description="Train T2S model")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
    parser.add_argument('--dataset_name', '-d', type=str, choices=['deadlift', 'benchpress'], help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=2000, help='training epochs')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='denoiser model save path')
    parser.add_argument('--subject', type=str, choices=['isolated','mix'], help='subject type')

    # model specific
    parser.add_argument('--general_seed', type=int, default=2025, help='seed for random number generation')
    parser.add_argument('--usepretrainedvae', default=True, help='pretrained vae')
    parser.add_argument('--total_step', type=int, default=100, help='sampling from [0,1]')

    # CLIP-specific arguments
    parser.add_argument('--use_clip', action='store_true',
                        help='Use CLIP-aligned text encoder instead of pre-stored embeddings')
    parser.add_argument('--clip_model_path', type=str, default=None,
                        help='Path to Stage 1 CLIP model checkpoint')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='Path to pretrained VAE (Stage 2 checkpoint)')

    args = parser.parse_args()
    args = get_cfg(args)

    if args.use_clip:
        if not args.clip_model_path:
            raise ValueError("--clip_model_path is required when --use_clip is set")

    if not args.pretrained_model_path:
        vae_dir = f'clip_{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}' if args.use_clip else f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}'
        args.pretrained_model_path = os.path.join('./results/saved_pretrained_models/', vae_dir, 'final_model.pth')
    print('pretrained vae: ', args.pretrained_model_path)
    print('checkpoint path: ', args.checkpoint_path)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clip_tag = 'clip' if args.use_clip else 'legacy'
    args.save_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}_{}_{}_{}'.format(
        args.backbone, args.denoiser, args.dataset_name, args.train_caption, args.pretrained_epc, args.subject, clip_tag))
    args.config = os.path.join('.', 'config', args.dataset_name + '.yaml')
    return args

if __name__ == '__main__':
    args = get_args()
    seed_everything(args.general_seed)
    train(args)
    print("Training complete.")
