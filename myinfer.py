import argparse
import torch
from matplotlib import pyplot as plt
from model.denoiser.mlp import MLP
from model.denoiser.mytransformer import Transformer
from model.pretrained.myvqvae import vqvae
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
import imageio
import os
import numpy as np
import math
from pretrained_mylavae import plot_pca_tsne
import json
from utils import get_cfg, RearV_BenchpressAnimator, TopV_BenchpressAnimator, LateralV_BenchpressAnimator, FullViewBenchpressAnimator
from myevaluation import calculate_mse, normalize

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

def plot_side_by_side_comparison(args, x_1_list, x_t_list, mse_list, subjects_list):
    save_path = args.generation_save_path_result
    for i in range(len(x_1_list)):
        fig_path = os.path.join(save_path, f'{subjects_list[i]}.jpg')
        plt.clf()
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{subjects_list[i]} {mse_list[i]:.4f}', fontsize=10)

        # 左圖：ground truth
        ax1 = plt.subplot(1, 2, 1)
        for j in range(len(x_1_list[i])):
            ax1.plot(x_1_list[i][j], label=f"{args.features[j]}")
        ax1.set_title('Ground Truth')
        ax1.legend()

        # 右圖：generated
        ax2 = plt.subplot(1, 2, 2)
        for j in range(len(x_t_list[i])):
            ax2.plot(x_t_list[i][j], label=f"{args.features[j]}")
        ax2.set_title('Generated')
        ax2.legend()

        plt.tight_layout()  # 留出 suptitle 空間
        plt.savefig(fig_path)
        plt.close()
        
def save_result(root, features):
    # save predict sample
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, f'data.json')
    full_animation = os.path.join(root, f'full_view.gif')
    FullViewBenchpressAnimator(features).animate(full_animation)
    with open(json_path, 'w') as f:
        json.dump(features, f, indent=4)

def infer(args):
    step = args.total_step
    cfg_scale = args.cfg_scale
    device = args.device

    print(f"Inference config::Step: {step}\t CFG Scale: {cfg_scale}")
    if args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    elif args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
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
    frames_list = []
    x_infer_list = []
    subjects_list = []
    with (torch.no_grad()):
        for batch, data in enumerate(test_loader):
            features = {feat : {} for feat in args.features[-args.input_dim:]}
            print(f'Generating {batch}th Batch TS...')

            y, x_1, embedding, subject = data
            y_list.append(y)
            print(y)
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
                    frames_list.append(xt_decode_np.copy())
                x_t_latent_dec = x_t.clone()
            
            x_t, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
            if batch == 0:
                x_t_infer_gt, after = pretrained_model.decoder(x_t_latent_enc, length=x_1.shape[-1])
                x_t_infer_gt = x_t_infer_gt.detach().cpu().numpy().squeeze()
                x_infer_list.append(x_t_infer_gt[0])

            x_1 = x_1.detach().cpu().numpy().squeeze()
            x_t = x_t.detach().cpu().numpy().squeeze()
            
            mse = calculate_mse(np.expand_dims(normalize(x_1), 0), np.expand_dims(normalize(x_t), 0))
            mse_list.append(mse)
            print(f'Batch {batch} MSE: {mse}')
            
            x_1_list.append(x_1)
            x_t_list.append(x_t)
            subjects_list.append(subject[0])
            
            for i, key in enumerate(features.keys()):
                features[key] = x_t[i].astype(float).tolist()
            save_path = os.path.join(args.generation_save_path_result, f'{subject[0]}')
            os.makedirs(save_path, exist_ok=True)
            save_result(save_path, features)
            np.save(os.path.join(save_path, f'x_t.npy'), x_t)
            if batch == 2:
                break
            
    plot_side_by_side_comparison(args, x_1_list, x_t_list, mse_list, subjects_list)
    plot_pca_tsne(x_1_list, x_t_list, args.generation_save_path_result)
    return x_1_list, subjects_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference flow matching model")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
    
    parser.add_argument('--cfg_scale', type=int, default=3, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='total step sampled from [0,1]')

    # for inference
    parser.add_argument('--checkpoint_id', type=int, default=2500,help='model id')
    parser.add_argument('--dataset_name', type=str, choices=['deadlift', 'benchpress'], help='dataset name')
    parser.add_argument('--run_time', type=int, default=1, help='inference run time')
    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name +'.yaml')
    args = get_cfg(args)
    args.pretrainedvae_path = os.path.join('./results/saved_pretrained_models', f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}', 'final_model.pth')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.checkpoint_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}_{}_{}_1D_patch'.format(args.backbone, args.denoiser, args.dataset_name, args.caption, args.pretrained_epc), 'model_{}.pth'.format(args.checkpoint_id))
    args.generation_save_path = os.path.join(args.save_path, 'generation', '{}_{}_{}_{}_{}_1D_patch'.format(args.backbone, args.denoiser, args.dataset_name, args.cfg_scale, args.total_step))
    
    print('pretrained vae path: ', args.pretrainedvae_path)
    print('checkpoint path: ', args.checkpoint_path)
    best_result = {}
    for i in range(args.run_time):
        args.generation_save_path_result = os.path.join(args.generation_save_path, f'run_{i}')
        x_1_list, subjects_list = infer(args)
    
        # save sample
        # The instruction implies that FullViewBenchpressAnimator should be called once for all results,
        # not per batch within the save_result function.
        # save sample
        features = {feat : {} for feat in args.features[-args.input_dim:]}
        for batch, x_1 in enumerate(x_1_list):
            for i, key in enumerate(features.keys()):
                features[key] = x_1[i].astype(float).tolist()
            full_animation = os.path.join(args.generation_save_path_result, f'{subjects_list[batch]}.gif')
            FullViewBenchpressAnimator(features).animate(full_animation)
