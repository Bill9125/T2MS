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
import time
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(loss_list, save_path, filename='loss_curve.png'):
    if len(loss_list) == 0:
        print("loss_list is empty, skipping plotting.")
        return
    plt.figure(figsize=(10, 6))
    x = [i/1000 for i in range(len(loss_list))]
    plt.plot(x, loss_list, label='Training Loss')
    plt.xlabel('Iteration(K)')
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
        
def train(args):
    print(f"Training config::\tepoch: {args.epochs}\tsave_path: {args.save_path}\tdevice: {args.device}")
    os.makedirs(args.save_path, exist_ok=True)
    train_loader, valid_loader, test_loader = loader_provider(args)
    model = {'DiT': Transformer, 'MLP': MLP}.get(args.denoiser)
    if model:
        model = model().to(args.device)
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
        
    ##################################################
    #                   mix train                  #
    ##################################################
    if args.mix_train:
        print("Mix training...")
        for epoch in range(start_epoch, args.epochs):
            loss = dict
            for i, (data1, data2, data3) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")):
                batch = i
                data = (data1, data2, data3)
                
                
                
                for j, data in enumerate(data):
                    y_text, x_1, y_text_embedding = data  # y_text:encoded text, x_1: Time Series Data, y_text_embedding: embeded text(using OpenAI embedding API)
                    # print(x_1.shape)
                    if x_1 == None:
                        continue
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
                    loss_list.append(loss.item())
                    optimizer.step()
                scheduler.step()
            print(f'[Epoch {epoch}] loss: {loss.item()}')
            
            if epoch % 100 == 0 or epoch == args.epochs - 1:
                print(f'Saving model {epoch} to {args.save_path}...')
                plot_loss_curve(loss_list, args.save_path)
                save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, loss_list=loss_list)
                torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))
                
            if epoch == 4000:
                break

            
    else:
        ##################################################
        #                   split train                  #
        ##################################################
        print("Not mix training...")
        for epoch in range(start_epoch, args.epochs):
            for batch, data in enumerate(train_loader):
                y_text, x_1, y_text_embedding = data  # y_text:encoded text, x_1: Time Series Data, y_text_embedding: embeded text(using OpenAI embedding API)
                y_text_embedding = y_text_embedding.float().to(args.device)
                x_1 = x_1.float().to(args.device)
                x_1,before = model.encoder(x_1)  # TS data ==>VAE==> clear TS embedding

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
                decide = torch.rand(1) < 0.3  #  for classifier free guidance
                if decide:
                    y_text_embedding = None
                pred = model(input=x_t, t=t, text_input=y_text_embedding)
                loss = backbone.loss(pred, noise_gt)
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                if batch % 100 == 0:
                    print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            scheduler.step()
            if epoch % 1000 == 0 or epoch == args.epochs - 1:
                print(f'Saving model {epoch} to {args.save_path}...')
                save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                                 loss_list=loss_list)
                torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))

def get_args():
    parser = argparse.ArgumentParser(description="Train T2S model")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
    parser.add_argument('--dataset_root', type=str, default='./Data', help='dataset root')
    parser.add_argument('--dataset_name', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--caption', type=str, default='Caption_with_feature_explanation')
    parser.add_argument('--pretrained_model_path', type=str, default='./results/saved_pretrained_models/36_benchpress_data_epoch75000/final_model.pth')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--epochs', type=int, default=20000, help='training epochs')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='denoiser model save path')
    parser.add_argument('--mix_train', type=bool, default=True, help='mixture train or not')
    parser.add_argument('--split_base_num', type=int, default=36)
    
    # model specific
    parser.add_argument('--general_seed', type=int, default=41, help='seed for random number generation')
    parser.add_argument('--usepretrainedvae', default=True, help='pretrained vae')
    parser.add_argument('--total_step', type=int, default=100, help='sampling from [0,1]')
    parser.add_argument('--backbone', type=str, default='flowmatching', help='flowmatching or ddpm or edm')
    parser.add_argument('--denoiser',type=str, default='DiT', help='DiT or MLP')
    
    # vae specific
    parser.add_argument('--block_hidden_size', type=int, default=128, help='hidden size of the blocks in the network')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers in the model')
    parser.add_argument('--res_hidden_size', type=int, default=256, help='hidden size of the residual layers')
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of the embeddings')
    parser.add_argument('--flow_dim', type=int, default=300, help='embedding dim flow into diffusion')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mix_train:
        args.data_length = 0
    args.save_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name, args.caption))
    return args

if __name__ == '__main__':
    args = get_args()
    seed_everything(args.general_seed)
    stime = time.time()
    train(args)
    etime = time.time()
    print(etime - stime)
