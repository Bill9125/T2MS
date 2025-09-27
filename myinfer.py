import argparse
import torch
from matplotlib import pyplot as plt
from model.denoiser.mlp import MLP
from model.denoiser.mytransformer import Transformer
from model.pretrained.myvqvae import vqvae
from datafactory.benchpress_dataloader import loader_provider
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from matplotlib.animation import FuncAnimation
import os
import numpy as np
import math
import random
from pretrained_mylavae import plot_pca_tsne

def plot_side_by_side_comparison(x_1, x_t, y_list, args):
    for i in range(len(x_1)):
        fig_path = os.path.join(args.generation_save_path, f'fig_{i}.jpg')
        plt.clf()
        plt.figure(figsize=(12, 6))
        plt.suptitle('comparison', fontsize=10)

        # 左圖：ground truth
        ax1 = plt.subplot(1, 2, 1)
        for j in range(len(x_1[i])):
            ax1.plot(x_1[i][j], label=f"ground truth {j}")
        ax1.set_title('Ground Truth')
        ax1.legend()

        # 右圖：generated
        ax2 = plt.subplot(1, 2, 2)
        for j in range(len(x_t[i])):
            ax2.plot(x_t[i][j], label=f"generated {j}")
        ax2.set_title('Generated')
        ax2.legend()

        plt.tight_layout()  # 留出 suptitle 空間
        plt.savefig(fig_path)
        plt.close()
        
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

def infer(args):
    step = args.total_step
    cfg_scale = args.cfg_scale
    generation_save_path_result = args.generation_save_path_result
    usepretrainedvae = args.usepretrainedvae
    device = args.device

    print(f"Inference config::Step: {step}\t CFG Scale: {cfg_scale}\t Use Pretrained VAE: {usepretrainedvae}")
    os.makedirs(generation_save_path_result, exist_ok=True)
    train_loader, valid_loader, test_loader = loader_provider(args, period='test')
    print('dataset length:', len(test_loader))
    vae = vqvae(args).to(device).float().eval()
    state = torch.load('results/saved_pretrained_models/36_benchpress_epoch75000/final_model.pth', map_location=device)  # 多半是 state_dict
    vae.load_state_dict(state)  # 正確載入
    pretrained_model = vae
    model = {'DiT': Transformer, 'MLP': MLP}.get(args.denoiser)
    if model:
        model = model().to(args.device)
    else:
        raise ValueError(f"No denoiser found")
    model.encoder = pretrained_model.encoder
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
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
    x_t_latent_enc_list = []
    x_t_latent_dec_list = []
    x_infer_list = []
    with (torch.no_grad()):
        for batch, data in enumerate(test_loader):
            print(f'Generating {batch}th Batch TS...')

            y, x_1, embedding = data
            y_list.append(y)
            x_1 = x_1.float().to(device)
            embedding = embedding.float().to(device)

            x_t, before = model.encoder(x_1)
            x_t_latent_enc = x_t.clone()
            x_t = torch.randn_like(x_t).float().to(device)
            for j in range(step):
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

                if batch == 0:
                    x_t_infer_stat, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
                    x_t_infer_stat = x_t_infer_stat.detach().cpu().numpy().squeeze()
                    x_infer_list.append(x_t_infer_stat[0])
            x_t_latent_dec = x_t.clone()
            x_t, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
            if batch == 0:
                x_t_infer_gt, after = pretrained_model.decoder(x_t_latent_enc, length=x_1.shape[-1])
                x_t_infer_gt = x_t_infer_gt.detach().cpu().numpy().squeeze()
                x_infer_list.append(x_t_infer_gt[0])

            x_1 = x_1.detach().cpu().numpy().squeeze()
            x_t = x_t.detach().cpu().numpy().squeeze()
            x_1_list.append(x_1)
            x_t_list.append(x_t)

            x_t_latent_dec = x_t_latent_dec.detach().cpu().numpy().squeeze()
            x_t_latent_enc = x_t_latent_enc.detach().cpu().numpy().squeeze()

            x_t_latent_dec_list.append(x_t_latent_dec)
            x_t_latent_enc_list.append(x_t_latent_enc)
            # if batch == 1:
            #     break
    
    x_1 = x_1_list
    x_t = x_t_list
    
    # x_1_array = np.concatenate(x_1_list, axis=1)
    # x_t_array = np.concatenate(x_t_list, axis=1)

    # x_t_latent_dec_array = np.concatenate(x_t_latent_dec_list, axis=0)
    # x_t_latent_enc_array = np.concatenate(x_t_latent_enc_list, axis=0)

    # x_1 = x_1_array[:, :, np.newaxis]
    # x_t = x_t_array[:, :, np.newaxis]
    # np.save(os.path.join(generation_save_path_result, 'x_1.npy'), x_1)
    # np.save(os.path.join(generation_save_path_result, 'x_t.npy'), x_t)
    # np.save(os.path.join(generation_save_path_result, 'x_t_latent_dec_array.npy'), x_t_latent_dec_array)
    # np.save(os.path.join(generation_save_path_result, 'x_t_latent_enc_array.npy'), x_t_latent_enc_array)

    return x_1, x_t, None, None, x_infer_list, y_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference flow matching model")
    # vae specific
    parser.add_argument('--block_hidden_size', type=int, default=128, help='hidden size of the blocks in the network')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers in the model')
    parser.add_argument('--res_hidden_size', type=int, default=256, help='hidden size of the residual layers')
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of the embeddings')
    parser.add_argument('--flow_dim', type=int, default=300, help='embedding dim flow into diffusion')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
    parser.add_argument('--usepretrainedvae', default=True, help='pretrained vae')
    parser.add_argument('--backbone', type=str, default='flowmatching', help='flowmatching or DDPM or EDM')
    parser.add_argument('--denoiser', type=str, default='DiT', help='DiT or MLP')
    parser.add_argument('--cfg_scale', type=float, default=7, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='total step sampled from [0,1]')
    parser.add_argument('--general_seed', type=int, default=41, help='seed for random number generation')

    # for inference
    parser.add_argument('--checkpoint_id', type=int, default=4000,help='model id')
    parser.add_argument('--dataset_root', type=str, default='./Data', help='dataset root')
    parser.add_argument('--dataset_name', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--caption', type=str, default='Caption')
    parser.add_argument('--split_base_num', type=int, default=36)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_root_path = args.dataset_name.split('_')[0]
    args.checkpoint_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}'.format(args.backbone, args.denoiser, model_root_path, args.caption), 'model_{}.pth'.format(args.checkpoint_id))
    args.generation_save_path = os.path.join(args.save_path, 'generation', '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,args.cfg_scale,args.total_step))

    args.generation_save_path_result = os.path.join(args.generation_save_path)
    x_1, x_t, _, _, x_infer_list, y_list = infer(args)

    plt.clf()
    plot_side_by_side_comparison(x_1, x_t, y_list, args)
    plot_pca_tsne(x_1, x_t, args.generation_save_path_result)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(x_infer_list[0]))
    line, = ax.plot(x, x_infer_list[0], color="cornflowerblue", lw=3)
    fixed_line, = ax.plot(x, x_infer_list[-1], color="black", lw=2, label='static line')
    ax.set_ylim(-0.2, 400)

    def init():
        line.set_ydata([np.nan] * len(x))
        return line, fixed_line


    def update(frame):
        idx = frame if frame < len(x_infer_list) else len(x_infer_list) - 1
        line.set_ydata(x_infer_list[idx])
        return line, fixed_line


    ani = FuncAnimation(fig, update, init_func=init, frames=150, interval=500, blit=True)
    ani.save(f"animation_{args.backbone}.gif", fps=15, writer="imageMagick")
