import os
import datetime
import json
import glob
import numpy as np
import argparse
import torch
import copy
from evaluate.utils import show_with_start_divider, show_with_end_divider, write_json_data, convert_numpy
from utils import get_cfg
from evaluate.visualization import generate_heatmaps, plot_tsne
from evaluate.data import load_vae_model, load_training_data, load_generated_data
from evaluate.evaluator import evaluate_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate flow matching model")
    parser.add_argument('--method_list', type=str, default='C-FID,NND,LDS', help='metric list [C-FID, NND, TSNE, LDS]')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for latent density score')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Save path')
    parser.add_argument('--dataset_name', '-d', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--cfg_scale', type=int, default=1, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='Total sampling steps')
    parser.add_argument('--run_time', type=int, default=10, help='Number of runs')
    parser.add_argument('--fixed_noise', action='store_true', help='Evaluate fixed random noise dataset')
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds to split the evaluation results to get standard deviation')
    parser.add_argument('--subject', type=str, choices=['isolated','mix'], help='subject type')
    parser.add_argument('--caption', type=str, choices=['explain','style_new'], help='caption type for inference')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 解析 method_list 為清單
    if not isinstance(args.method_list, list):
        args.method_list = [m.strip() for m in args.method_list.split(',')]
        
    # 設定路徑
    args.pretrainedvae_path = os.path.join('./results/saved_pretrained_models', 
                                         f'{args.split_base_num}_{args.dataset_name}_epoch{args.pretrained_epc}_{args.subject}', 
                                         'final_model.pth')
    args.model_name = f'{args.backbone}_{args.denoiser}_{args.dataset_name}_{args.cfg_scale}_{args.total_step}_{args.subject}_{args.caption}'
    noise_type = "fixed" if getattr(args, 'fixed_noise', False) else "random"
    args.generation_save_path = os.path.join(args.save_path, f'generation_{noise_type}', args.model_name)
    args.evaluation_save_path = os.path.join(args.save_path, f'evaluation_{noise_type}', args.model_name)

    # 載入模型與資料
    vae_encoder = load_vae_model(args)
    train_repr_my, train_repr_fid_my, train_embs_my, train_labels_my = load_training_data(args, vae_encoder)
    x_1_list, x_t_list, emb_list, grouped_samples, gen_labels = load_generated_data(args)

    result = {}
    max_len = max(x.shape[-1] for x in x_t_list)
    
    # 1. 全域評估 (Global Marginals)
    ori_data_arr = np.array([np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), 'constant') for x in x_1_list])
    gen_data_arr = np.array([np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), 'constant') for x in x_t_list])
    all_embs_arr = np.concatenate(emb_list, axis=0)
    
    print(f'Original data shape: {ori_data_arr.shape}, Generated data shape: {gen_data_arr.shape}')
    result, all_ori_repr, all_gen_repr = evaluate_data(args, ori_data_arr, gen_data_arr, 'all_samples', result, vae_encoder=vae_encoder, train_repr_my=train_repr_my, cond_embs=all_embs_arr)
    
    if 'TSNE' in args.method_list:
        # Check if the other caption's generated data exists to plot both in t-SNE
        other_caption = "style_new" if args.caption == "explain" else "explain"
        other_model_name = f'{args.backbone}_{args.denoiser}_{args.dataset_name}_{args.cfg_scale}_{args.total_step}_{args.subject}_{other_caption}'
        other_generation_save_path = os.path.join(args.save_path, f'generation_{noise_type}', other_model_name)
        
        other_args = copy.deepcopy(args)
        other_args.generation_save_path = other_generation_save_path
        
        other_gen_repr = None
        if os.path.exists(other_generation_save_path) and glob.glob(os.path.join(other_generation_save_path, "run_*")):
            print(f"Found other caption's generated data in: {other_generation_save_path}")
            try:
                _, other_x_t_list, _, _, other_gen_labels = load_generated_data(other_args)
                if other_x_t_list:
                    other_max_len = max(x.shape[-1] for x in other_x_t_list)
                    other_gen_data_arr = np.array([np.pad(x, ((0, 0), (0, other_max_len - x.shape[-1])), 'constant') for x in other_x_t_list])
                    with torch.no_grad():
                        other_gen_tensor = torch.tensor(other_gen_data_arr).float().to(args.device)
                        other_gen_features, _ = vae_encoder(other_gen_tensor)
                        other_gen_repr = other_gen_features.mean(dim=-1).cpu().numpy()
            except Exception as e:
                print(f"Error loading/processing other caption's generated data: {e}")
        else:
            print(f"Other caption's generated data not found in: {other_generation_save_path}")

        tsne_save_path = os.path.join('.', 'heatmaps', args.subject, f"tsne_{args.model_name}_{noise_type}.png")
        
        # If both are available, plot both with correct labels, otherwise fall back to single set
        if other_gen_repr is not None:
            kwargs = {
                args.caption: all_gen_repr,
                f"{args.caption}_labels": gen_labels,
                other_caption: other_gen_repr,
                f"{other_caption}_labels": other_gen_labels
            }
            plot_tsne(all_ori_repr, None, tsne_save_path, "(All Samples)", train_repr=train_repr_fid_my, **kwargs)
        else:
            plot_tsne(all_ori_repr, all_gen_repr, tsne_save_path, "(All Samples)", train_repr=train_repr_fid_my, gen_labels=gen_labels)

    # 2. 獨立分類評估 (Class-Conditional Evaluation)
    print("\n--- Running Class-Conditional Evaluation ---")
    for error_class, data_dict in grouped_samples.items():
        if len(data_dict['x_t']) <= 1:
            print(f"Skipping class [{error_class}] - Not enough samples.")
            continue
            
        print(f"Evaluating Class: [{error_class}] ({len(data_dict['x_t'])} samples)")
        class_max_len = max(x.shape[-1] for x in data_dict['x_t'])
        class_ori_arr = np.array([np.pad(x, ((0, 0), (0, class_max_len - x.shape[-1])), 'constant') for x in data_dict['x_1']])
        class_gen_arr = np.array([np.pad(x, ((0, 0), (0, class_max_len - x.shape[-1])), 'constant') for x in data_dict['x_t']])
        class_embs_arr = np.concatenate(data_dict['emb'], axis=0)
        
        class_train_repr = None
        if train_repr_my is not None:
            mask = np.array([error_class in str(label) for label in train_labels_my])
            if np.sum(mask) > 0:
                class_train_repr = train_repr_my[mask]
            else:
                print(f"Warning: No training samples found matching class '{error_class}'. Using full train set as fallback.")
                class_train_repr = train_repr_my

        result, _, _ = evaluate_data(args, class_ori_arr, class_gen_arr, f'class_{error_class}', result, vae_encoder=vae_encoder, train_repr_my=class_train_repr, cond_embs=class_embs_arr)
    print("--------------------------------------------\n")

    # 彙整與儲存結果
    if isinstance(result, dict) and result:
        summary = {}
        for key in result:
            for metric, value in result[key].items():
                summary[metric] = summary.get(metric, 0) + value
        result['summary'] = {metric: val / len(result) for metric, val in summary.items()}
        result = convert_numpy(result)
        
        # 額外儲存完整的 Namespace config 參數，以便隨時查閱實驗條件
        result['config'] = convert_numpy(vars(args))
        
        os.makedirs(args.evaluation_save_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(args.evaluation_save_path, f'{args.model_name}_{timestamp}.json')
        write_json_data(result, save_path)
        print(f'Evaluation results saved to {save_path}')
        
        generate_heatmaps(args, result)
        
    show_with_end_divider(f'Evaluation done. Results: {result}')

