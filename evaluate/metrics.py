import argparse
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def calculate_mrr(ori_data, gen_data, k=None):
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
            if similarities[idx] > therehold:
                rank = idx + 1
                break

        mrr_scores[batch_idx] = 1.0 / rank if rank is not None else 0.0

    return np.mean(mrr_scores)

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

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE)
    y_true, y_pred: shape = (N, T) or (T,) for time series
    """
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    return numerator / denominator if denominator != 0 else np.inf

def mrr_at_10(y_true: np.ndarray, y_gen: np.ndarray, threshold=0.9) -> float:
    """
    Mean Reciprocal Rank at 10
    y_true: shape = (N, D)    # N: samples, D: feature dimension
    y_gen: shape = (N, 10, D) # N: samples, 10 generated candidates
    """
    N = y_true.shape[0]
    reciprocal_ranks = []

    for i in range(N):
        # 計算 yi_gen_0 ~ yi_gen_9 和 yi 的 cosine similarity
        sims = cosine_similarity(y_gen[i], y_true[i].reshape(1, -1)).flatten()  # shape: (10,)
        
        # 找第一個超過 threshold 的 index（+1 因為 rank 是從 1 開始）
        relevant = np.where(sims > threshold)[0]
        if len(relevant) > 0:
            reciprocal_ranks.append(1.0 / (relevant[0] + 1))
        else:
            reciprocal_ranks.append(0.0)  # 找不到 relevant 結果

    return np.mean(reciprocal_ranks)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)

def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    計算資料的相關矩陣（Pearson correlation），輸入 shape = (N, T, D)
    輸出 shape = (D, D)
    """
    N, T, D = data.shape
    data_reshaped = data.reshape(N * T, D)  # 合併樣本與時間
    corr_matrix = np.corrcoef(data_reshaped, rowvar=False)  # shape = (D, D)
    return corr_matrix

def calculate_correlational_score(real_data: np.ndarray, gen_data: np.ndarray) -> float:
    """
    計算 Correlational Score（Ni et al., 2020）
    real_data, gen_data: shape = (N, T, D)
    """
    C_real = calculate_correlation_matrix(real_data)
    C_gen = calculate_correlation_matrix(gen_data)

    numerator = np.sum(np.abs(C_real - C_gen))
    denominator = np.sum(np.abs(C_real))
    
    if denominator == 0:
        return np.nan  # 防止除以 0

    score = 1 - (numerator / denominator)
    return score

def calculate_dtw(seq1, seq2, dist_fn=None):
    """
    計算兩個序列之間的 DTW (Dynamic Time Warping) 距離

    Args:
        seq1 (np.ndarray): shape (T1, D) 或 (T1,)
        seq2 (np.ndarray): shape (T2, D) 或 (T2,)
        dist_fn (function): 距離函數，預設為歐氏距離平方

    Returns:
        float: 最小 DTW 距離
    """
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    
    if dist_fn is None:
        # 預設為歐氏距離平方
        dist_fn = lambda x, y: np.sum((x - y)**2)

    T1 = len(seq1)
    T2 = len(seq2)
    dtw = np.full((T1 + 1, T2 + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = dist_fn(seq1[i - 1], seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],    # insertion
                                    dtw[i, j - 1],    # deletion
                                    dtw[i - 1, j - 1])  # match
    
    return np.sqrt(dtw[T1, T2])  # 如果你希望最後回傳的是根號距離

def mean_std(data):
    data = np.array(data)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std

def min_max_normalize_columns(data):
    """將每一欄 (column) 正規化到 [0, 1] 區間"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # 避免除以 0
    normalized = (data - min_vals) / ranges
    return normalized

def zscore_normalize_columns(data):
    """將每一欄 (column) 進行 Z-score 標準化"""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1  # 避免除以 0
    normalized = (data - means) / stds
    return normalized

def calculate_sequence_correlation(ori_batch, gen_batch):
    """
    處理批次資料的序列相關性計算
    
    Parameters:
    - ori_batch: shape (batch_size, T, D) 的原始序列批次
    - gen_batch: shape (batch_size, T, D) 的生成序列批次
    
    Returns:
    - results: 每個批次的 (最佳位移, 最小距離) 列表
    """
    batch_size = ori_batch.shape[0]
    for b in range(batch_size):
        # 提取單一批次的序列
        ori_seq = ori_batch[b]  # shape: (T, D)
        gen_seq = gen_batch[b]  # shape: (T, D)
        
        # 計算該批次的相關性
        best_shift, min_dist = sequence_correlation(ori_seq, gen_seq)
    
    return best_shift, min_dist

def sequence_correlation(seq_a, seq_b, max_shift=None):
    """
    計算兩個多特徵序列的相關性
    
    Parameters:
    - seq_a: shape (m, d) 的多特徵序列
    - seq_b: shape (n, d) 的多特徵序列  
    - max_shift: 最大位移量，預設為較短序列長度
    
    Returns:
    - best_shift: 最佳位移量
    - min_distance: 最小平均歐幾里得距離
    - all_distances: 所有位移的距離記錄
    """
    m, n = len(seq_a), len(seq_b)
    
    if max_shift is None:
        max_shift = min(m, n) - 1
    
    distances = {}
    
    # 遍歷所有可能的位移
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            # 正向位移：B向右移
            overlap_len = min(m, n - shift)
            if overlap_len <= 0:
                continue
            a_aligned = seq_a[:overlap_len]
            b_aligned = seq_b[shift:shift + overlap_len]
        else:
            # 負向位移：B向左移（等同A向右移）
            overlap_len = min(m + shift, n)
            if overlap_len <= 0:
                continue
            a_aligned = seq_a[-shift:-shift + overlap_len]
            b_aligned = seq_b[:overlap_len]
        
        # 計算對應點的歐幾里得距離
        euclidean_distances = np.linalg.norm(a_aligned - b_aligned, axis=1)
        mean_distance = np.mean(euclidean_distances)
        distances[shift] = mean_distance
    
    # 找到最小距離及對應位移
    best_shift = min(distances.keys(), key=lambda k: distances[k])
    min_distance = distances[best_shift]
    
    return best_shift, min_distance

def plt_metrics(scores, output_path, met):
    mean, std = mean_std(scores)
    indices = list(range(len(mean)))

    # 📊 繪圖
    plt.clf()
    plt.figure(figsize=(10, 6))
    # DTW
    plt.errorbar(indices, mean, yerr=std, fmt='o-', label=met, color='blue', capsize=5)
    for x, y, std in zip(indices, mean, std):
        plt.text(x, y + std + 0.01, f'{y:.3f}±{std:.3f}', ha='center', color='blue', fontsize=9)

        
    plt.xlabel("Generated File Index (merged_i.txt)")
    plt.ylabel("Score")
    plt.title("metrics on same subject")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    rival = {2: 'recording_20241226_161846', 3: 'recording_20241226_162038',
                    4: 'recording_20241226_162307', 5: 'recording_20241226_162449'}
    parser.add_argument('--rival', default=1, type=int)
    parser.add_argument('--sport', choices=['benchpress', 'deadlift'], type=str)
    args = parser.parse_args()

    root = './3D_Real_Final/Category_1/recording_20241226_161701'
    if args.rival == 1:
        rival_root = root
        size = 9
    else:
        rival_root = f'./3D_Real_Final/Category_{args.rival}/{rival[args.rival]}'
        size = 10

    # 建立分數記錄清單
    process_type = ['_delta_', '_delta2_', '_delta_square_', '_']
    metrics = {'mse':[], 'wape':[], 'correlational_score':[], 'dtw':[], 'sequence_correlation':[]}
    ori_data = []
    gen_data = []
    

    for type in process_type:
        all_mse_scores = np.empty((0, size))
        all_wape_scores = np.empty((0, size))
        all_corr_scores = np.empty((0, size))
        all_dtw_scores = np.empty((0, size))
        all_min_distances = np.empty((0, size))
        for i in range(1, 11):
            with open(os.path.join(root, f'filtered{type}norm/merged_{i}.txt'), 'r', encoding="utf-8") as f:
                lines = f.read().strip().split('\n')
                ori_data = np.array([list(map(float, line.split(','))) for line in lines])
                ori_data = min_max_normalize_columns(ori_data)  # shape: (T, D)

            mse_scores = np.array([])
            wape_scores = np.array([])
            corr_scores = np.array([])
            
            dtw_scores = np.array([])
            min_distances = np.array([])

            for j in range(1, 11):
                if args.rival == 1:
                    if j == i:
                        continue
                with open(os.path.join(rival_root, f'filtered{type}norm/merged_{j}.txt'), 'r', encoding="utf-8") as f:
                    lines = f.read().strip().split('\n')
                    gen_data = np.array([list(map(float, line.split(','))) for line in lines])
                    gen_data = min_max_normalize_columns(gen_data)  # shape: (T, D)

                # 增加 batch 維度，讓 shape = (1, T, D)
                ori_data_batch = ori_data[np.newaxis, :, :]
                gen_data_batch = gen_data[np.newaxis, :, :]

                mse_score = calculate_mse(ori_data_batch, gen_data_batch)
                wape_score = calculate_wape(ori_data_batch, gen_data_batch)
                correlational_score = calculate_correlational_score(ori_data_batch, gen_data_batch)
                dtw_score = calculate_dtw(ori_data_batch, gen_data_batch)
                best_shift, min_distance = calculate_sequence_correlation(ori_data_batch, gen_data_batch)
                
                mse_scores = np.concatenate([mse_scores, np.array([mse_score])], axis=0)
                wape_scores = np.concatenate([wape_scores, np.array([wape_score])], axis=0)
                corr_scores = np.concatenate([corr_scores, np.array([correlational_score])], axis=0)
                dtw_scores = np.concatenate([dtw_scores, np.array([dtw_score])], axis=0)
                min_distances = np.concatenate([min_distances, np.array([min_distance])], axis=0)

            all_mse_scores = np.vstack([all_mse_scores, mse_scores])
            all_wape_scores = np.vstack([all_wape_scores, wape_scores])
            all_corr_scores = np.vstack([all_corr_scores, corr_scores])
            all_dtw_scores = np.vstack([all_dtw_scores, dtw_scores])
            all_min_distances = np.vstack([all_min_distances, min_distances])
            
        metrics = {'mse':all_mse_scores, 'wape':all_wape_scores, 'correlational_score':all_corr_scores, 'dtw':all_dtw_scores, 'sequence_correlation':all_min_distances}
        for metric, val in metrics.items():
            output_dir = f'./metrics_test/exp{args.rival}/{metric}'
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'./metrics_test/exp{args.rival}/{metric}/{type}result.jpg'
            plt_metrics(val, output_path, metric)
    
    print(dtw_score, ori_data_batch, gen_data_batch)