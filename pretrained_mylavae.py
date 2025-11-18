import argparse
import numpy as np
import os
import torch
from model.pretrained.myvqvae import vqvae
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import seed_everything, plot_loss_curve, get_cfg

def plot_comparison_animation(real, recon, save_dir, gif_name='comparison.gif', fps=2):
    """
    real / recon  : List[np.ndarray]，每筆 shape 必須為 [13, T]
    save_dir      : 目錄
    """
    if len(real) == 0:
        print('無資料可視覺化')
        return

    # 建圖：1×2 欄，左 = 所有 Real 13 條；右 = 所有 Recon 13 條
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # 為 10 條曲線各建一條 Line2D
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 10, endpoint=False)]
    lines_real = [axL.plot([], [], c=colors[i], lw=1.2, label=f'f{i}')[0] for i in range(10)]
    lines_reco = [axR.plot([], [], c=colors[i], lw=1.2, label=f'f{i}')[0] for i in range(10)]

    # 標題、圖例
    axL.set_title('Real');  axR.set_title('Reconstructed')
    axL.legend(fontsize=7, ncol=1, loc='upper right')
    axR.legend(fontsize=7, ncol=1, loc='upper right')

    # 文字標註時間長度
    txt_L = axL.text(0.02, 0.92, '', transform=axL.transAxes, fontsize=9)
    txt_R = axR.text(0.02, 0.92, '', transform=axR.transAxes, fontsize=9)

    def init():
        for ln in lines_real + lines_reco:
            ln.set_data([], [])
        txt_L.set_text(''); txt_R.set_text('')
        return lines_real + lines_reco + [txt_L, txt_R]

    def update(idx):
        r = real[idx]      # [10, T1]
        z = recon[idx]     # [10, T2]

        # 每條特徵曲線分別更新
        for i in range(7):
            xr = np.arange(r.shape[1]);  xz = np.arange(z.shape[1])
            lines_real[i].set_data(xr, r[i])
            lines_reco[i].set_data(xz, z[i])

        # 動態調整座標界線（統一兩邊 y 軸方便對比）
        ymin = min(r.min(), z.min());  ymax = max(r.max(), z.max())
        pad  = 0.05 * (ymax - ymin + 1e-8)
        axL.set_xlim(0, r.shape[1] - 1);     axL.set_ylim(ymin - pad, ymax + pad)
        axR.set_xlim(0, z.shape[1] - 1);     axR.set_ylim(ymin - pad, ymax + pad)

        # 更新標註
        txt_L.set_text(f'T={r.shape[1]}')
        txt_R.set_text(f'T={z.shape[1]}')

        fig.suptitle(f'Sample #{idx}', fontsize=12)
        return lines_real + lines_reco + [txt_L, txt_R]

    ani = animation.FuncAnimation(
        fig, update, frames=len(real),
        init_func=init, blit=True,
        interval=int(1000 / max(1, fps)), repeat=True
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, gif_name)
    ani.save(out_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f'GIF saved to {out_path}')

def flatten_sample(x_np):
    # 將 [n_f, T] 或 [T] 攤平成 1D 特徵向量用於 PCA/TSNE。
    return x_np.reshape(-1)

def plot_pca_tsne(real_samples, reconstructed_samples, save_path):
    # 檢查並處理不同形狀的樣本
    def safe_flatten_and_pad(samples):
        flattened = [flatten_sample(x) for x in samples]
        max_len = max(len(f) for f in flattened)
        
        # 將所有樣本 pad 到相同長度
        padded = []
        for f in flattened:
            if len(f) < max_len:
                padded_f = np.pad(f, (0, max_len - len(f)), mode='constant', constant_values=0)
            else:
                padded_f = f
            padded.append(padded_f)
        
        return np.stack(padded, axis=0)
    
    real_flat = safe_flatten_and_pad(real_samples)
    reco_flat = safe_flatten_and_pad(reconstructed_samples)
    
    combined = np.vstack((real_flat, reco_flat))
    labels = ['Real'] * len(real_flat) + ['Reconstructed'] * len(reco_flat)

    # PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)

    # t-SNE
    n = combined.shape[0]
    perplexity = max(2, min(n - 1, 30))
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto')
    combined_tsne = tsne.fit_transform(combined)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=combined_pca[:, 0], y=combined_pca[:, 1], hue=labels, ax=axs[0])
    axs[0].set_title('PCA')
    sns.scatterplot(x=combined_tsne[:, 0], y=combined_tsne[:, 1], hue=labels, ax=axs[1])
    axs[1].set_title('t-SNE')
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/pca_tsne.png")
    plt.close()

def inference(model, test_loader, device, save_dir, num_samples=None):
    model.eval()
    real_samples = []
    reconstructed_samples = []
    with torch.no_grad():
        seen_batches = 0
        for batch in test_loader:
            # batch 是 List[(texts, xs, embeddings, gt_cat)]，逐組處理
            for (texts, xs, embeddings, _) in batch:
                xs = xs.float().to(device)  # [B, n_f, T]
                loss, recon_error, reconstructed, z = model.shared_eval(xs, None, mode='test')
                
                # 轉 numpy 並收集
                real_np = xs.detach().cpu().numpy()
                reco_np = reconstructed.detach().cpu().numpy()
                
                # 收集所有樣本而不是逐一繪圖
                for b in range(real_np.shape[0]):
                    real_samples.append(real_np[b])
                    reconstructed_samples.append(reco_np[b])
                    
                seen_batches += 1
                if num_samples is not None and seen_batches >= num_samples:
                    break
            if num_samples is not None and seen_batches >= num_samples:
                break

    # 在最後一次性生成動畫
    if len(real_samples) > 0 and len(reconstructed_samples) > 0:
        plot_comparison_animation(real_samples, reconstructed_samples, save_dir, fps=1)
        plot_pca_tsne(real_samples, reconstructed_samples, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['deadlift', 'benchpress'], help='dataset name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='results/saved_pretrained_models/', help='denoiser model save path')
    parser.add_argument('--only_inference', type=bool, default=False)

    # Model-specific parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument('--config', type=str, default='config.yaml', help='model configuration')
    parser.add_argument('--compression_factor', type=int, default=4, help='compression factor')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost used in the loss function')
    args = get_cfg(parser.parse_args())

    save_folder_name = '{}_{}_epoch{}'.format(args.split_base_num, args.dataset_name, args.pretrained_epc)
    save_dir = os.path.join(args.save_path, save_folder_name)
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.general_seed)

    model = vqvae(args).to(device)
    optimizer = model.configure_optimizers(lr=args.learning_rate)

    # 取得 train/test 的 DataLoader（內含自訂 collate_fn，會回傳分組子批次 list）
    if args.dataset_name == 'deadlift':
        from datafactory.deadlift.dataloader import loader_provider
    elif args.dataset_name == 'benchpress':
        from datafactory.benchpress.dataloader import loader_provider
    train_loader, test_loader = loader_provider(args)

    if not args.only_inference:
        total_epochs = int((args.pretrained_epc / max(1, len(train_loader))) + 0.5)
        print(f'total epoch : {total_epochs}')
        loss_list = []
        for epoch in range(total_epochs):
            model.train()
            epoch_losses = []
            group_losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
                for (texts, xs, embeddings, _) in batch:
                    xs = xs.clone().detach().float().to(device)  # [B_g, n_f, T]
                    loss, recon_error, x_recon, z = model.shared_eval(xs, optimizer, 'train')
                    group_losses.append(loss.item())
                if len(group_losses) > 0:
                    epoch_losses.append(np.mean(group_losses))
            print(f"Epoch: {epoch+1}, Batch: {len(epoch_losses)}, Loss: {np.mean(group_losses):.6f}")
            loss_list.append(np.mean(group_losses))
            # 週期性儲存
            if epoch % max(1, total_epochs // 10) == 0:
                plot_loss_curve(loss_list, save_dir, filename=f'loss_curve_epoch.png')
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
                print(f'Saved Model from epoch: {epoch}')


        torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        print("Training complete.")

    print("Starting inference...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'final_model.pth'), map_location=device))
    inference(model, test_loader, device, save_dir, num_samples=None)
