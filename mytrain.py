import argparse
import os
import torch
from torch.optim import AdamW, lr_scheduler
from datafactory.benchpress_dataloader import loader_provider
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from model.denoiser.mytransformer import Transformer
from model.denoiser.mlp import MLP
from model.pretrained.myvqvae import vqvae
from tqdm import tqdm
from utils import plot_loss_curve, seed_everything
import numpy as np
from utils import get_cfg

def train(args):
    print(f"Training config::\tepoch: {args.epochs}\tsave_path: {args.save_path}\tdevice: {args.device}")
    os.makedirs(args.save_path, exist_ok=True)
    train_loader, test_loader = loader_provider(args)
    model = {'DiT': Transformer(args.flow_dim), 'MLP': MLP}.get(args.denoiser)
    if model:
        model = model.to(args.device)
    else:
        raise ValueError(f"No denoiser found")

    pretrained_model = vqvae(args).float().to(args.device)
    pretrained_model.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device(args.device)))
    # pretrained_model = torch.load(args.pretrained_model_path, map_location=torch.device(args.device), weights_only=False)
    # pretrained_model.float().to(args.device)
    backbone = {'flowmatching': RectifiedFlow(), 'ddpm': DDPM(args.total_step, args.device)}.get(args.backbone)
    if backbone:
        pass
    else:
        raise ValueError(f"No backbone found")

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
    # if from checkpoint:
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
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
            for (y_text, x_1, y_text_embedding) in group:
                y_text_embedding = y_text_embedding.float().to(args.device)
                x_1 = x_1.float().to(args.device)
                x_1, before = model.encoder(x_1)  # TS data ==>VAE==> clear TS embedding

                if args.backbone == 'flowmatching':
                    t = torch.round(torch.rand(x_1.size(0), device=args.device) * args.total_step) / args.total_step
                    x_t, x_0 = backbone.create_flow(x_1, t)  # x_t: dirty TS embedding, x_0：pure noise
                    noise_gt = x_1 - x_0
                elif args.backbone == 'ddpm':
                    t = torch.floor(torch.rand(x_1.size(0)).to(args.device) * args.total_step).long()
                    noise_gt = torch.randn_like(x_1).float().to(args.device)
                    x_t, n_xt = backbone.q_sample(x_1, t, noise_gt)
                else:
                    raise ValueError(f"Unsupported backbone type: {args.backbone}")

                optimizer.zero_grad()
                decide = torch.rand(1) < 0.3
                if decide:
                    y_text_embedding = None
                pred = model(input=x_t, t=t, text_input=y_text_embedding)
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
            save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, loss_list=loss_list)
            torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))

        if epoch == 7000:
            break

def get_args():
    parser = argparse.ArgumentParser(description="Train T2S model")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
    parser.add_argument('--dataset_name', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--pretrained_model_path', type=str, default='./results/saved_pretrained_models/36_benchpress_epoch80000/final_model.pth')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=20000, help='training epochs')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='denoiser model save path')

    # model specific
    parser.add_argument('--general_seed', type=int, default=2025, help='seed for random number generation')
    parser.add_argument('--usepretrainedvae', default=True, help='pretrained vae')
    parser.add_argument('--total_step', type=int, default=100, help='sampling from [0,1]')
    parser.add_argument('--config', type=str, default='config.yaml', help='model configuration')
    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.save_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name, args.caption, '80000'))
    return args

if __name__ == '__main__':
    args = get_args()
    seed_everything(args.general_seed)
    train(args)
    print("Training complete.")
