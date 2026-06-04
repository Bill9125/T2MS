"""
Stage 1: CLIP Contrastive Learning — Text Encoder + LA Encoder Alignment

Trains the LA Encoder and CLIP projection heads (text + latent) using symmetric
InfoNCE loss so that text embeddings and time-series latents share a common
embedding space.

The Text Encoder here takes pre-computed text embeddings (128-dim from
caption.json) as input and projects them into the shared CLIP space.

Usage:
    python train_clip.py -d benchpress --subject mix
    python train_clip.py -d benchpress --subject mix --epochs 500
"""

import argparse
import os
import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm

from model.pretrained.myvqvae import vqvae
from model.pretrained.clip_heads import LAEncoderCLIPHead, clip_loss
from utils import get_cfg, seed_everything, plot_loss_curve


def train_clip(args):
    print(f"=== Stage 1: CLIP Contrastive Training ===")
    print(f"  dataset:     {args.dataset_name}")
    print(f"  subject:     {args.subject}")
    print(f"  clip_dim:    {args.clip_dim}")
    print(f"  temperature: {args.clip_temperature}")
    print(f"  epochs:      {args.epochs}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  device:      {args.device}")

    os.makedirs(args.save_path, exist_ok=True)

    # --- Data ---
    if args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    elif args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    train_loader, test_loader = loader_provider(args)

    # --- Models ---
    # LA Encoder (from vqvae, but we only need the encoder)
    vae_model = vqvae(args).float().to(args.device)

    # Initialize from pretrained VAE checkpoint
    if args.pretrained_vae_path and args.pretrained_vae_path.lower() != 'none' and os.path.exists(args.pretrained_vae_path):
        print(f"  Initializing VAE from pretrained checkpoint: {args.pretrained_vae_path}")
        vae_state = torch.load(args.pretrained_vae_path, map_location=args.device, weights_only=False)
        vae_model.load_state_dict(vae_state)
    else:
        if args.pretrained_vae_path and args.pretrained_vae_path.lower() == 'none':
            print("  Initializing VAE encoder from scratch (random initialization) explicitly.")
        else:
            print(f"  Warning: Pretrained VAE path not found: {args.pretrained_vae_path}. Training VAE encoder from scratch!")

    la_encoder = vae_model.encoder  # Encoder module

    # Freeze VAE encoder if requested
    if args.freeze_encoder:
        for param in la_encoder.parameters():
            param.requires_grad = False
        print("  VAE Encoder: FROZEN (only training projection heads)")
    else:
        print("  VAE Encoder: TRAINABLE (fine-tuning)")

    # Detect pre-computed text embedding dimension from a dummy batch
    dummy_batch = next(iter(train_loader))
    text_emb_dim = dummy_batch[0][2].shape[-1]
    print(f"  Dynamically detected pre-computed text embedding dimension: {text_emb_dim}")
    
    # We no longer use TextEncoder to project text embeddings. 
    # Instead, we set clip_dim equal to the text embedding dimension,
    # and project motion features directly to match this space.
    args.clip_dim = text_emb_dim

    # LA Encoder CLIP projection head
    la_clip_head = LAEncoderCLIPHead(
        embedding_dim=args.embedding_dim,
        clip_dim=text_emb_dim,
    ).to(args.device)

    # --- Optimizer: train LA Encoder (if not frozen) + LA projection head ---
    trainable_params = []
    if not args.freeze_encoder:
        trainable_params += list(la_encoder.parameters())
    trainable_params += list(la_clip_head.parameters())
    
    total_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"  Total trainable parameters: {total_trainable}")

    optimizer = AdamW(trainable_params, lr=args.clip_lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Training loop ---
    loss_history = []
    val_loss_history = []
    start_epoch = 0

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        la_encoder.load_state_dict(checkpoint['la_encoder'])
        if 'text_encoder' in checkpoint and checkpoint['text_encoder'] is not None:
            print("  Warning: Loaded checkpoint has text_encoder, but it is ignored in the new no-TextEncoder pipeline.")
        la_clip_head.load_state_dict(checkpoint['la_clip_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        print(f"  Resumed from epoch {start_epoch}")

    la_encoder.train()
    la_clip_head.train()

    print("Training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            for group in batch:
                texts, xs, embeddings, subjects, clips = group

                # embeddings: [B, 128] pre-computed text embeddings
                embeddings = embeddings.float().to(args.device)
                xs = xs.float().to(args.device)

                # Filter out zero embeddings (missing captions)
                valid_mask = embeddings.abs().sum(dim=-1) > 0
                if valid_mask.sum() < 2:
                    continue  # Need at least 2 samples for contrastive loss

                valid_xs = xs[valid_mask]
                valid_embeddings = embeddings[valid_mask]

                # Forward: LA Encoder → CLIP head
                z, _ = la_encoder(valid_xs)            # [B, embed_dim, flow_dim]
                la_emb = la_clip_head(z)               # [B, clip_dim]

                # Direct Alignment: use the pre-extracted text embedding directly
                text_emb = valid_embeddings            # [B, clip_dim]

                # CLIP loss
                loss = clip_loss(text_emb, la_emb, temperature=args.clip_temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        loss_history.append(avg_loss)

        # Run validation every epoch (very fast!)
        val_loss = validate_clip(test_loader, la_encoder, la_clip_head, args)
        val_loss_history.append(val_loss)

        print(f"[Epoch {epoch}] CLIP loss: {avg_loss:.5f}  val_loss: {val_loss:.5f}  lr: {scheduler.get_last_lr()[0]:.2e}")

        # Periodic save
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            save_checkpoint(args, epoch, la_encoder, text_emb_dim, la_clip_head,
                            optimizer, loss_history, val_loss_history=val_loss_history)
            plot_loss_curve(loss_history, args.save_path, filename='clip_loss_curve.png', val_loss_list=val_loss_history)

    # Final save
    save_checkpoint(args, args.epochs - 1, la_encoder, text_emb_dim, la_clip_head,
                    optimizer, loss_history, val_loss_history=val_loss_history, filename='final_clip_model.pth')
    plot_loss_curve(loss_history, args.save_path, filename='clip_loss_curve.png', val_loss_list=val_loss_history)
    print("Stage 1 (CLIP) training complete.")


def save_checkpoint(args, epoch, la_encoder, text_emb_dim, la_clip_head,
                    optimizer, loss_history, val_loss_history=None, filename=None):
    if filename is None:
        filename = f'clip_model_epoch_{epoch}.pth'
    save_dict = {
        'epoch': epoch,
        'la_encoder': la_encoder.state_dict(),
        'la_clip_head': la_clip_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history if val_loss_history is not None else [],
        'clip_dim': args.clip_dim,
        'embedding_dim': args.embedding_dim,
        'text_emb_dim': text_emb_dim,
    }
    path = os.path.join(args.save_path, filename)
    torch.save(save_dict, path)
    print(f"  Saved checkpoint to {path}")



@torch.no_grad()
def validate_clip(test_loader, la_encoder, la_clip_head, args):
    la_encoder.eval()
    la_clip_head.eval()

    losses = []
    for batch in test_loader:
        for group in batch:
            texts, xs, embeddings, subjects, clips = group

            embeddings = embeddings.float().to(args.device)
            xs = xs.float().to(args.device)

            valid_mask = embeddings.abs().sum(dim=-1) > 0
            if valid_mask.sum() < 2:
                continue

            valid_xs = xs[valid_mask]
            valid_embeddings = embeddings[valid_mask]

            z, _ = la_encoder(valid_xs)
            la_emb = la_clip_head(z)
            text_emb = valid_embeddings

            loss = clip_loss(text_emb, la_emb, temperature=args.clip_temperature)
            losses.append(loss.item())

    la_encoder.train()
    la_clip_head.train()

    return np.mean(losses) if losses else float('nan')


def get_args():
    parser = argparse.ArgumentParser(description="Stage 1: CLIP Contrastive Training")
    parser.add_argument('--dataset_name', '-d', type=str, choices=['deadlift', 'benchpress'],
                        required=True, help='dataset name')
    parser.add_argument('--subject', type=str, default='mix', choices=['isolated', 'mix'],
                        help='subject split type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='training epochs (overrides config)')
    parser.add_argument('--save_path', type=str, default='./results/clip_models/',
                        help='checkpoint save path')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='resume from checkpoint')
    parser.add_argument('--pretrained_vae_path', type=str, default=None,
                        help='path to pretrained VAE weights to initialize encoder')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='freeze the VAE encoder and only train projection heads')

    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fallback to default pretrained VAE checkpoint path if not provided
    if not args.pretrained_vae_path:
        args.pretrained_vae_path = os.path.join(
            './results/saved_pretrained_models/',
            f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}',
            'final_model.pth'
        )

    # Use config defaults if not overridden by CLI
    if args.epochs is None:
        args.epochs = args.clip_epoch
    if args.batch_size == 128:
        args.batch_size = args.clip_batch_size

    args.save_path = os.path.join(
        args.save_path,
        f'{args.dataset_name}_{args.subject}_dim{args.clip_dim}'
    )
    return args


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.general_seed)
    train_clip(args)
