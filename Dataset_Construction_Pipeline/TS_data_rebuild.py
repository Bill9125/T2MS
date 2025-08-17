"""
This script is part of the dataset construction pipeline for time series data.
It processes samples, generates summaries, and saves them in JSON format.
Merge all vision skeleton data into one file for caption generation.
"""
import os
import os.path as path
import json
import glob
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

def sort_files(data):
    # 儲存尾數對應的每個類別的檔案路徑
    grouped = defaultdict(dict)

    # 建立尾數到檔案路徑的映射
    for key, paths in data.items():
        for path in paths:
            filename = os.path.basename(path)
            match = re.search(r'_(\d+)\.txt$', filename)
            if match:
                suffix = match.group(1)  # 抓取尾數
                grouped[int(suffix)][key] = path  # 這邊轉成 int 方便後面排序

    # 將尾數相同的 group zip 起來（只保留同時擁有所有 keys 的）
    all_keys = set(data.keys())
    zipped = []

    for suffix in sorted(grouped.keys()):  # 按數字順序處理每個尾數
        file_dict = grouped[suffix]
        if set(file_dict.keys()) == all_keys:
            # 依據 key 排序以確保順序一致
            zipped.append(tuple(file_dict[k] for k in sorted(all_keys)))
    
    return zipped

def merge_files(subject_path, output_root):
    lengths = []
    for subject in subject_path:
        recordings = glob.glob(path.join(subject, '*'))
        recordings = [r for r in recordings if not r.endswith('_棋盤')]
        for recording in recordings:
            if path.basename(recording) == 'recording_20241223_174625':
                continue
            info = {'angle': [], 'angle_vision1': [], 'angle_vision5': [], 'bar': []}
            output_dir = path.join(output_root, path.basename(subject), path.basename(recording))
            if not path.exists(output_dir):
                os.makedirs(output_dir)

            folder = glob.glob(path.join(recording, '*'))
            folder = [f for f in folder if path.basename(f) in list(info.keys())]
            for f in folder:  # f is the info folder
                txt_files = glob.glob(path.join(f, '*.txt'))
                info[path.basename(f)] = txt_files

            zip_data = sort_files(info)
            arrange = {}
            for i, z in enumerate(zip_data):
                tmp = []
                for txt in z:
                    data = np.loadtxt(txt, delimiter=",")
                    data = data[:, 1:]
                    if path.basename(path.dirname(txt)) == 'bar':
                        data = data[:, 0:2]
                    tmp.append(data)
                arrange[i] = np.hstack(tmp).tolist()
                if len(arrange[i]) < 50 or len(arrange[i]) > 250:
                    print(f"Warning: Sample {i+1} in {path.basename(subject)}/{path.basename(recording)} has an unexpected length: {len(arrange[i])}")
                    continue
                lengths.append(len(arrange[i]))
                json.dump(arrange, open(path.join(output_dir, 'data.json'), 'w'), indent=4)
    return lengths

if __name__ == "__main__":
    output_root = './Data/Deadlift'
    data_path = './Data/recordings_copy'
    subject_path = glob.glob(path.join(data_path, '*'))
    subject_path = [p for p in subject_path if path.isdir(p)]
    lengths = merge_files(subject_path, output_root)
    
    data = np.array(lengths)
    # print('mean:', np.mean(data), 'std:', np.std(data), 'median:', np.median(data), 'max:', np.max(data), 'min:', np.min(data))
    
    # plt.hist(lengths, bins=5)
    # plt.xlabel('Length')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Data Lengths')
    # plt.savefig(path.join(output_root, 'length_distribution.png'))
    
    