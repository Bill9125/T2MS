"""
Stage 2: LA Decoder Training — With Frozen CLIP-aligned LA Encoder

Trains the LA Decoder to reconstruct time-series data from the latent
representations produced by the frozen CLIP-aligned LA Encoder (from Stage 1).

Usage:
    python train_decoder.py -d benchpress --subject mix \\
        --clip_model_path ./results/clip_models/benchpress_mix_dim256/final_clip_model.pth
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm

from model.pretrained.myvqvae import vqvae
from utils import get_cfg, seed_everything, plot_loss_curve


def train_decoder(args):
    print(f"=== Stage 2: LA Decoder Training (Frozen Encoder) ===")
    print(f"  dataset:         {args.dataset_name}")
    print(f"  subject:         {args.subject}")
    print(f"  clip_model_path: {args.clip_model_path}")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  device:          {args.device}")

    os.makedirs(args.save_path, exist_ok=True)

    # --- Data ---
    if args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    elif args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    train_loader, test_loader = loader_provider(args)

    # --- Model ---
    model = vqvae(args).float().to(args.device)

    # Load CLIP-aligned LA Encoder weights from Stage 1
    clip_ckpt = torch.load(args.clip_model_path, map_location=args.device, weights_only=False)
    model.encoder.load_state_dict(clip_ckpt['la_encoder'])
    print(f"  Loaded LA Encoder from Stage 1 checkpoint")

    # Freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    print(f"  LA Encoder: FROZEN")

    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"  Decoder trainable parameters: {decoder_params}")

    # --- Optimizer ---
    optimizer = AdamW(model.decoder.parameters(), lr=args.learning_rate, weight_decay=0.0)

    # --- Training loop ---
    loss_history = []
    val_loss_history = []
    start_epoch = 0

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        print(f"  Resumed from epoch {start_epoch}")

    # Determine total epochs based on the same logic as pretrained_mylavae.py
    total_epochs = args.epochs

    print("Training...")
    for epoch in range(start_epoch, total_epochs):
        model.decoder.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
            for (texts, xs, embeddings, subjects, clips) in batch:
                xs = xs.float().to(args.device)  # [B, n_f, T]
                L = xs.shape[-1]

                # Forward (encoder is frozen)
                with torch.no_grad():
                    z, before = model.encoder(xs)  # z: [B, E, flow_dim], before: [B, E, T/4]

                data_recon, after = model.decoder(z, length=L)  # recon: [B, n_f, T]

                recon_error = F.mse_loss(data_recon, xs)
                cross_loss = F.mse_loss(before, after)
                loss = recon_error + cross_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        loss_history.append(avg_loss)

        # Run validation every epoch (very fast!)
        val_loss = validate_decoder(model, test_loader, args)
        val_loss_history.append(val_loss)

        print(f"[Epoch {epoch}] loss: {avg_loss:.6f}  recon: {recon_error.item():.6f}  val_loss: {val_loss:.6f}")

        # Periodic save & validation
        if epoch % max(1, total_epochs // 10) == 0 or epoch == total_epochs - 1:
            plot_loss_curve(loss_history, args.save_path, filename='decoder_loss_curve.png', val_loss_list=val_loss_history)
            save_dict = {
                'epoch': epoch,
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
            }
            torch.save(save_dict, os.path.join(args.save_path, f'decoder_epoch_{epoch}.pth'))
            print(f"  Saved decoder checkpoint at epoch {epoch}")

    # Final save: save the full vqvae model (encoder + decoder) for Stage 3
    final_path = os.path.join(args.save_path, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"  Saved final full model (encoder+decoder) to {final_path}")
    print("Stage 2 (Decoder) training complete.")


    # Run PCA/t-SNE and comparison animation on the test set
    print("Running Stage 2 visualization on test set...")
    run_decoder_visualization(model, test_loader, args)


@torch.no_grad()
def run_decoder_visualization(model, test_loader, args):
    model.eval()
    real_samples = []
    reconstructed_samples = []

    for batch in test_loader:
        for (texts, xs, embeddings, subjects, clips) in batch:
            xs = xs.float().to(args.device)  # [B, n_f, T]
            L = xs.shape[-1]

            z, _ = model.encoder(xs)
            data_recon, _ = model.decoder(z, length=L)

            real_np = xs.detach().cpu().numpy()
            reco_np = data_recon.detach().cpu().numpy()

            for b in range(real_np.shape[0]):
                real_samples.append(real_np[b])
                reconstructed_samples.append(reco_np[b])

    if len(real_samples) > 0 and len(reconstructed_samples) > 0:
        print("Generating decoder comparison animation and PCA/t-SNE plots...")
        from pretrained_mylavae import plot_comparison_animation, plot_pca_tsne
        plot_comparison_animation(real_samples, reconstructed_samples, args.save_path, fps=1)



@torch.no_grad()
def validate_decoder(model, test_loader, args):
    model.decoder.eval()
    losses = []

    for batch in test_loader:
        for (texts, xs, embeddings, subjects, clips) in batch:
            xs = xs.float().to(args.device)
            L = xs.shape[-1]

            z, before = model.encoder(xs)
            data_recon, after = model.decoder(z, length=L)

            recon_error = F.mse_loss(data_recon, xs)
            cross_loss = F.mse_loss(before, after)
            loss = recon_error + cross_loss
            losses.append(loss.item())

    return np.mean(losses) if losses else float('nan')


def get_args():
    parser = argparse.ArgumentParser(description="Stage 2: LA Decoder Training")
    parser.add_argument('--dataset_name', '-d', type=str, choices=['deadlift', 'benchpress'],
                        required=True, help='dataset name')
    parser.add_argument('--subject', type=str, default='mix', choices=['isolated', 'mix'],
                        help='subject split type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='training epochs (defaults to vae.epoch from config)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--save_path', type=str, default='./results/saved_pretrained_models/',
                        help='decoder save path')
    parser.add_argument('--clip_model_path', type=str, required=True,
                        help='path to Stage 1 CLIP model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='resume decoder training from checkpoint')

    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.epochs is None:
        # Same total-update logic as pretrained_mylavae.py
        args.epochs = 200  # Will be refined after seeing data loader length

    args.save_path = os.path.join(
        args.save_path,
        f'clip_{args.split_base_num}_{args.dataset_name}_epoch{args.epochs}_{args.subject}'
    )
    return args


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.general_seed)
    train_decoder(args)
