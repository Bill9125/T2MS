from pydoc import text
from torch.utils.data import Dataset, DataLoader, random_split
from .dataset import BenchpressT2SDataset
import torch
import numpy as np
import os
import json
import random

class AlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.datasets = [dataset1, dataset2, dataset3]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.total_length = sum(self.lengths)
        self.index_map = {}
        offset = 0
        for i, length in enumerate(self.lengths):
            for j in range(length):
                self.index_map[offset + j] = (i, j)
            offset += length
    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        dataset_idx, sub_idx = self.index_map[index]
        return self.datasets[dataset_idx][sub_idx], dataset_idx

def custom_collate_fn(batch):
    # 檢查是否為 AlternatingDataset 格式 ( (data), dataset_idx )
    # AlternatingDataset 回傳的 tuple 長度為 2，而原始 Dataset 回傳的 data tuple 長度為 5
    if not (isinstance(batch[0], tuple) and len(batch[0]) == 2 and isinstance(batch[0][1], int)):
        # 如果是原始 Dataset，將其包裝成 dataset_idx=0 的格式以相容後續邏輯
        batch = [(item, 0) for item in batch]

    grouped_data = {0: [], 1: [], 2: []}
    grouped_data = {
        idx: [data for data, dataset_idx in batch if dataset_idx == idx]
        for idx in grouped_data.keys()
    }
    batches = []
    for idx, data_list in grouped_data.items():
        if data_list:
            batch_texts, batch_xs, batch_embeddings, subjects, clips = zip(*data_list)
            batch_texts = [torch.from_numpy(text) if isinstance(text, np.ndarray) else text for text in batch_texts]
            batch_subjects = [torch.from_numpy(subject) if isinstance(subject, np.ndarray) else subject for subject in subjects]
            batch_xs = torch.stack(
                [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in batch_xs]
            )
            batch_embeddings = torch.stack(
                [torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding for embedding in batch_embeddings]
            )
            batch_clips = [torch.from_numpy(clip) if isinstance(clip, np.ndarray) else clip for clip in clips]
            batches.append((batch_texts, batch_xs, batch_embeddings, batch_subjects, batch_clips))
    return batches

def loader_provider(args, period='train'):
    # --- 1. 定義切分參數 ---
    r_train, r_test = (0.7, 0.3)
    gen = torch.Generator().manual_seed(args.general_seed)
    json_path = os.path.join(args.dataset_root, args.dataset_name, 'data.json')
    
    def get_caption_dir(c):
        return c if c.startswith('Caption_') else f'Caption_{c}'
        
    caption_root_train = os.path.join(args.dataset_root, args.dataset_name, get_caption_dir(args.train_caption))
    caption_root_test = os.path.join(args.dataset_root, args.dataset_name, get_caption_dir(args.caption))
    
    # --- 2. 決定受試者過濾清單 (僅在 Isolated 模式下需要) ---
    train_subs, test_subs = None, None
    if args.subject == 'isolated':
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        all_subjects = sorted(list(all_data.keys()))
        random.seed(args.general_seed)
        random.shuffle(all_subjects)
        n_train = int(r_train * len(all_subjects))
        train_subs = all_subjects[:n_train]
        test_subs = all_subjects[n_train:]

    # --- 3. 根據模式與週期建立 Dataset ---
    if period == 'train':
        # 訓練時期：通常需要回傳 Train Loader 與 Test Loader
        # 注意：在 Subject Isolated 模式下，Train/Test 讀取不同的受試者
        # 在 Subject Mix 模式下，先讀取全部受試者，後面再用 random_split
        curr_train_subs = train_subs if train_subs is not None else None
        
        ds1 = BenchpressT2SDataset(args.features, json_path, caption_root_train, 'train', emb_dim=128, data_dim=args.split_base_num, allowed_subjects=curr_train_subs)
        ds2 = BenchpressT2SDataset(args.features, json_path, caption_root_train, 'train', emb_dim=128, data_dim=args.split_base_num*2, allowed_subjects=curr_train_subs)
        ds3 = BenchpressT2SDataset(args.features, json_path, caption_root_train, 'train', emb_dim=128, data_dim=args.split_base_num*4, allowed_subjects=curr_train_subs)
        train_ds = AlternatingDataset(ds1, ds2, ds3)
        
        if train_subs is not None:
            # Isolated 模式：Test DS 直接讀取 Test 受試者
            test_ds = BenchpressT2SDataset(args.features, json_path, caption_root_train, 'test', emb_dim=128, data_dim=args.split_base_num*2, allowed_subjects=test_subs)
        else:
            # Mix 模式：目前 train_ds 包含全部資料，待會進行 random_split
            train_ds, test_ds = random_split(train_ds, [r_train, r_test], generator=gen)
            
        common = dict(batch_size=args.batch_size, collate_fn=custom_collate_fn)
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common)
        return train_loader, test_loader

    elif period == 'test':
        # 測試/推論時期：僅回傳 Test Loader
        curr_test_subs = test_subs if test_subs is not None else None
        
        if args.caption == 'style_new':
            curr_test_subs = None
            
        test_ds = BenchpressT2SDataset(args.features, json_path, caption_root_test, 'test', emb_dim=128, data_dim=args.split_base_num*2, allowed_subjects=curr_test_subs)
        
        if args.subject == 'mix' and args.caption != 'style_new':
            # Mix 模式：從全集中切出測試部分
            _, test_ds = random_split(test_ds, [r_train, r_test], generator=gen)
            
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, batch_size=args.batch_size, collate_fn=custom_collate_fn)
        return None, test_loader
    else:
        raise ValueError(f"Unknown period: {period}")

if __name__ == "__main__":
    pass