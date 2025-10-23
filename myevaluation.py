from itertools import count
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from Dataset_Construction_Pipeline.Evaluate_Datasets import cosine_similarity
from scipy.linalg import sqrtm
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from evaluate.ts2vec import initialize_ts2vec
from evaluate.feature_based_measures import calculate_mdd, calculate_acd, calculate_sd, calculate_kd
import os
import datetime
from evaluate.utils import show_with_start_divider, show_with_end_divider, determine_device, write_json_data
import argparse
import torch
from scipy.stats import norm
from utils import get_cfg

def normalize(x):
    # 沿每一列(axis=1)計算最小和最大值，keepdims=True 保持維度方便廣播
    min_val = x.min(axis=1, keepdims=True)
    max_val = x.max(axis=1, keepdims=True)
    # 計算縮放後的結果
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)  # 加 epsilon 避免除零錯誤
    return x_norm


###################################################
#                    MRR                          #
###################################################

def calculate_mrr(ori_data, gen_data, k=None):
    threshold = 0.5
    n_batch_size = ori_data.shape[0]
    n_generations = gen_data.shape[3]
    k = n_generations if k is None else k

    mrr_scores = np.zeros(n_batch_size)

    for batch_idx in range(n_batch_size):
        similarities = []
        for gen_idx in range(k):
            real_sequence = ori_data[batch_idx]
            generated_sequence = gen_data[batch_idx, :, :, gen_idx]
            similarity = cosine_similarity(real_sequence, generated_sequence)
            similarities.append(np.mean(similarity))

        sorted_indices = np.argsort(similarities)[::-1]
        rank = None
        for idx in sorted_indices:
            if similarities[idx] > threshold:
                rank = idx + 1
                break

        mrr_scores[batch_idx] = 1.0 / rank if rank is not None else 0.0

    return np.mean(mrr_scores)

###################################################
#             other reconstruct:CRPS              #
###################################################

def calculate_crps(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_timesteps = ori_data.shape[1]
    n_series = ori_data.shape[2]
    n_generations = gen_data.shape[3]
    crps_values = []

    for i in range(n_samples):
        total_crps = 0

        for j in range(n_series):
            crps_list = []

            for k in range(n_generations):
                mean = gen_data[i, :, j, k].mean()
                std_dev = gen_data[i, :, j, k].std()
                if std_dev == 0:
                    std_dev += 1e-8
                obs_value = ori_data[i, :, j]
                cdf_obs = np.where(obs_value < mean, 0, 1)

                cdf_pred = norm.cdf(obs_value, loc=mean, scale=std_dev)

                crps = np.mean((cdf_obs - cdf_pred) ** 2)
                crps_list.append(crps)

            average_crps = np.mean(crps_list)
            total_crps += average_crps

        crps_values.append(total_crps / n_series)

    crps_values = np.array(crps_values)
    average_crps = crps_values.mean()
    return average_crps


def evaluate_muldata(args, ori_data, gen_data):
    show_with_start_divider(f"Evalution with settings:{args}")

    # Parse configs
    method_list = args.method_list
    dataset_name = args.dataset_name
    model_name = args.model_name
    device = args.device
    evaluation_save_path = args.evaluation_save_path

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M%S")
    combined_name = f'{model_name}_{dataset_name}_{formatted_time}_multi'

    if not isinstance(method_list, list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]
    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None

    result = {}

    if 'CRPS' in method_list:
        mdd = calculate_crps(ori_data, gen_data)
        result['CRPS'] = mdd
    if 'MRR' in method_list:
        mrr = calculate_mrr(ori_data, gen_data)
        result['MRR'] = mrr

    if isinstance(result, dict):
        evaluation_save_path = os.path.join(evaluation_save_path, f'{combined_name}.json')
        write_json_data(result, evaluation_save_path)
        print(f'Evaluation denoiser_results saved to {evaluation_save_path}.')

    show_with_end_divider(f'Evaluation done. Results:{result}.')

    return result


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_ed(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)

    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu

def calculate_dtw(ori_data,comp_data):
    distance_dtw = []
    n_samples = ori_data.shape[0]
    for i in range(n_samples):
        distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
        distance_dtw.append(distance)

    distance_dtw = np.array(distance_dtw)
    average_distance_dtw = distance_dtw.mean()
    return average_distance_dtw

def calculate_mse(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    mse_values = []

    for i in range(n_samples):
        total_mse = 0
        for j in range(n_series):
            mse = np.mean((ori_data[i, :, j] - gen_data[i, :, j]) ** 2)
            total_mse += mse
        mse_values.append(total_mse / n_series)

    mse_values = np.array(mse_values)
    average_mse = mse_values.mean()
    return average_mse

def calculate_wape(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    wape_values = []

    for i in range(n_samples):
        total_absolute_error = 0
        total_actual_value = 0

        for j in range(n_series):
            absolute_error = np.abs(ori_data[i, :, j] - gen_data[i, :, j])
            total_absolute_error += np.sum(absolute_error)
            total_actual_value += np.sum(np.abs(ori_data[i, :, j]))

        if total_actual_value != 0:
            wape = total_absolute_error / total_actual_value
        else:
            wape = np.nan

        wape_values.append(wape)

    wape_values = np.array(wape_values)
    average_wape = np.nanmean(wape_values)
    return average_wape



def evaluate_data(args, ori_data, gen_data, index, result):
    show_with_start_divider(f"Evalution with settings:{args}")

    # Parse configs
    method_list = args.method_list
    device = args.device

    if not isinstance(method_list,list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]

    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None
    if ori_data.shape != gen_data.shape:
        print(f'Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}.')
        show_with_end_divider('Error: Generated data does not have the same shape with original data.')
        return None

    result[index] = {}
    if 'C-FID' in method_list:
        fid_model = initialize_ts2vec(np.transpose(ori_data, (0, 2, 1)),device)
        ori_repr = fid_model.encode(np.transpose(ori_data,(0, 2, 1)), encoding_window='full_series')
        gen_repr = fid_model.encode(np.transpose(gen_data,(0, 2, 1)), encoding_window='full_series')
        cfid = calculate_fid(ori_repr,gen_repr)
        result[index]['C-FID'] = cfid

    if 'MSE' in method_list:
        mse = calculate_mse(ori_data,gen_data)
        result[index]['MSE'] = mse
    if 'WAPE' in method_list:
        wape = calculate_wape(ori_data,gen_data)
        result[index]['WAPE'] = wape
    if 'MRR' in method_list:
        mrr = calculate_mrr(ori_data,gen_data)
        result[index]['MRR'] = mrr
    if 'CRPS' in method_list:
        crps = calculate_crps(ori_data,gen_data)
        result[index]['CRPS'] = crps
    if 'ED' in method_list:
        ed = calculate_ed(ori_data,gen_data)
        result[index]['ED'] = ed
    if 'ACD' in method_list:
        acd = calculate_acd(ori_data,gen_data)
        result[index]['ACD'] = acd
    if 'SD' in method_list:
        sd = calculate_sd(ori_data,gen_data)
        result[index]['SD'] = sd
    if 'KD' in method_list:
        kd = calculate_kd(ori_data,gen_data)
        result[index]['KD'] = kd
    if 'DTW' in method_list:
        dtw = calculate_dtw(ori_data,gen_data)
        result[index]['DTW'] = dtw

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flow matching model")
    parser.add_argument('--method_list', type=str, default='MSE,WAPE,MRR,DTW',
                            help='metric list [MSE,WAPE,MRR,CRPS,C-FID,ED,ACD,SD,KD,DTW]')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
    parser.add_argument('--config', type=str, default='config.yaml', help='model configuration')
    parser.add_argument('--dataset_name', type=str, default='benchpress', help='dataset name')
    parser.add_argument('--cfg_scale', type=float, default=3, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='total step sampled from [0,1]')
    parser.add_argument('--run_time', type=int, default=10, help='total run time')

    args = parser.parse_args()
    args = get_cfg(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_name = '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,args.cfg_scale, args.total_step)
    args.generation_save_path = os.path.join(args.save_path, 'generation',
                                            '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,
                                                                    args.cfg_scale, args.total_step))
    args.evaluation_save_path = os.path.join(args.save_path, 'evaluation', args.model_name)

    result = {}
    '''evaluate our model'''
    for sample in range(10):
        x_1_list = []
        x_t_list = []
        for j in range(args.run_time):
            args.generation_save_path_result = os.path.join(args.generation_save_path, f'run_{j}')
            print(os.path.join(args.generation_save_path_result, f'x_t_sample_{sample}.npy'))
            x_t = np.load(os.path.join(args.generation_save_path_result, f'x_t_sample_{sample}.npy'))
            x_1 = np.load(os.path.join(args.generation_save_path, f'x_1_sample_{sample}.npy'))
            x_t = normalize(x_t)
            x_1 = normalize(x_1)
            x_t_list.append(x_t)
            x_1_list.append(x_1)

        print(f'ori_data shape:{np.array(x_t_list).shape}, gen_data shape:{np.array(x_1_list).shape}')
        result = evaluate_data(args, np.array(x_t_list), np.array(x_1_list), sample, result)  # batch, dim , time length
    
    if isinstance(result, dict):
        summary = {}
        for key in result:
            for metric, value in result[key].items():
                summary[metric] = summary.get(metric, 0) + value
        for metric in summary:
            summary[metric] /= len(result)
        result['summary'] = summary
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y%m%d-%H%M%S")
        combined_name = f'{args.model_name}_{args.dataset_name}_{formatted_time}'
        evaluation_save_path = os.path.join(args.evaluation_save_path, f'{combined_name}.json')
        write_json_data(result, evaluation_save_path)
        print(f'Evaluation denoiser_results saved to {evaluation_save_path}.')
    
    show_with_end_divider(f'Evaluation done. Results:{result}.')
