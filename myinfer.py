import argparse
import torch
from matplotlib import pyplot as plt
from model.denoiser.mlp import MLP
from model.denoiser.mytransformer import Transformer
from model.pretrained.myvqvae import vqvae
from datafactory.benchpress_dataloader import loader_provider
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
import imageio
import os
import numpy as np
import math
from pretrained_mylavae import plot_pca_tsne
import json
from utils import get_cfg

def calculate_mse(ori_data, gen_data):
    mse_values = []
    for i in range(ori_data.shape[0]):
        mse = np.mean((ori_data[i] - gen_data[i]) ** 2)
        mse_values.append(mse)
    return np.mean(mse_values)

def save_diffusion_gif(frames, save_path, filename='diffusion.gif'):
    gif_path = os.path.join(save_path, filename)
    images = []
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots()
        if frame.ndim == 1:
            ax.plot(frame)
        else:
            for j in range(frame.shape[0]):
                ax.plot(frame[j])
        ax.set_title(f'Diffusion Step {100*i}')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    imageio.mimsave(gif_path, images, duration=0.5)  # 可調整 duration
    print(f'GIF saved to {gif_path}')

def plot_side_by_side_comparison(x_1, x_t, mse_list, save_path):
    for i in range(len(x_1)):
        fig_path = os.path.join(save_path, f'sample_{i}.jpg')
        plt.clf()
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Comparison {mse_list[i]:.4f}', fontsize=10)

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

def infer(args):
    step = args.total_step
    cfg_scale = args.cfg_scale
    device = args.device

    print(f"Inference config::Step: {step}\t CFG Scale: {cfg_scale}")
    _, test_loader = loader_provider(args, period='test')
    print('dataset length:', len(test_loader))
    vae = vqvae(args).to(device).float().eval()
    state = torch.load(args.pretrainedvae_path, map_location=device)  # 多半是 state_dict
    vae.load_state_dict(state)  # 正確載入
    pretrained_model = vae
    model = {'DiT': Transformer(args.flow_dim), 'MLP': MLP}.get(args.denoiser)
    if model:
        model = model.to(args.device)
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
    mse_list = []
    frames = []
    frames_list = []
    x_t_latent_enc_list = []
    x_t_latent_dec_list = []
    x_infer_list = []
    with (torch.no_grad()):
        for batch, data in enumerate(test_loader):
            features = {'left_shoulder_y': [], 'right_shoulder_y': [], 'left_dist': [], 'right_dist': [], 'left_elbow': [], 'left_shoulder': [], 'right_elbow': [], 'right_shoulder': [], 'left_torso-arm': [], 'right_torso-arm': []}
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
                    
                if (j % 100 == 0) or (j == step - 1):
                    xt_decode, _ = pretrained_model.decoder(x_t, length=x_1.shape[-1])
                    xt_decode_np = xt_decode.detach().cpu().numpy().squeeze()
                    frames.append(xt_decode_np.copy())
                x_t_latent_dec = x_t.clone()
            
            x_t, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
            if batch == 0:
                x_t_infer_gt, after = pretrained_model.decoder(x_t_latent_enc, length=x_1.shape[-1])
                x_t_infer_gt = x_t_infer_gt.detach().cpu().numpy().squeeze()
                x_infer_list.append(x_t_infer_gt[0])

            x_1 = x_1.detach().cpu().numpy().squeeze()
            x_t = x_t.detach().cpu().numpy().squeeze()
            for i, key in enumerate(features.keys()):
                features[key] = x_t[i].astype(float).tolist()
            json_path = os.path.join(args.generation_save_path_result, f'samples_{batch}.json')
            with open(json_path, 'w') as f:
                json.dump(features, f, indent=4)
            print(f'Features saved to {json_path}')
            
            mse = calculate_mse(x_1, x_t)
            mse_list.append(mse)
            print(f'Batch {batch} MSE: {mse}')
            x_1_list.append(x_1)
            x_t_list.append(x_t)
            if batch == 0:
                break
    return x_1_list, x_t_list, None, None, x_infer_list, y_list, frames, mse_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference flow matching model")
    parser.add_argument('--config', type=str, default='config.yaml', help='model configuration')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
    parser.add_argument('--pretrainedvae_path', default='./results/saved_pretrained_models/36_benchpress_epoch80000/final_model.pth', help='pretrained vae')
    parser.add_argument('--cfg_scale', type=int, default=3, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='total step sampled from [0,1]')

    # for inference
    parser.add_argument('--checkpoint_id', type=int, default=900,help='model id')
    parser.add_argument('--dataset_name', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--run_time', type=int, default=10, help='inference run time')
    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_root_path = args.dataset_name.split('_')[0]
    args.checkpoint_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, model_root_path, args.caption, '80000'), 'model_{}.pth'.format(args.checkpoint_id))
    args.generation_save_path = os.path.join(args.save_path, 'generation', '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,args.cfg_scale,args.total_step))

    x_t_path = os.path.join(args.generation_save_path, 'x_t.npy')
    best_result = {}
    for i in range(args.run_time):
        args.generation_save_path_result = os.path.join(args.generation_save_path, f'run_{i}')
        x_1_path = os.path.join(args.generation_save_path_result, 'x_1.npy')
        os.makedirs(args.generation_save_path_result, exist_ok=True)
        x_1, x_t, _, _, x_infer_list, y_list, frames_list, mse_list = infer(args)
        np.save(x_1_path, x_1)
        plot_side_by_side_comparison(x_1, x_t, mse_list, args.generation_save_path_result )
        # plot_pca_tsne(x_1, x_t, args.generation_save_path_result)
        # save_diffusion_gif(frames_list, args.generation_save_path, filename=f'diffusion.gif')
    np.save(x_t_path, x_t)
