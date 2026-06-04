from torch.utils.data import Dataset, DataLoader, random_split
from .dataset import DeadliftT2SDataset
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
    if not (isinstance(batch[0], tuple) and len(batch[0]) == 2 and isinstance(batch[0][1], int)):
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
        curr_train_subs = train_subs if train_subs is not None else None
        
        ds1 = DeadliftT2SDataset(args.features, json_path, caption_root_train, 'train', emb_dim=128, data_dim=args.split_base_num, allowed_subjects=curr_train_subs)
        ds2 = DeadliftT2SDataset(args.features, json_path, caption_root_train, 'train', emb_dim=128, data_dim=args.split_base_num*2, allowed_subjects=curr_train_subs)
        ds3 = DeadliftT2SDataset(args.features, json_path, caption_root_train, 'train', emb_dim=128, data_dim=args.split_base_num*4, allowed_subjects=curr_train_subs)
        train_ds = AlternatingDataset(ds1, ds2, ds3)
        
        if train_subs is not None:
            test_ds = DeadliftT2SDataset(args.features, json_path, caption_root_train, 'test', emb_dim=128, data_dim=args.split_base_num*2, allowed_subjects=test_subs)
        else:
            train_ds, test_ds = random_split(train_ds, [r_train, r_test], generator=gen)
            
        common = dict(batch_size=args.batch_size, collate_fn=custom_collate_fn)
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common)
        return train_loader, test_loader

    elif period == 'test':
        curr_test_subs = test_subs if test_subs is not None else None
        
        if args.caption == 'style_new':
            curr_test_subs = None
            
        data_dim_test = 0 if getattr(args, 'batch_size', 1) == 1 else args.split_base_num*2
        test_ds = DeadliftT2SDataset(args.features, json_path, caption_root_test, 'test', emb_dim=128, data_dim=data_dim_test, allowed_subjects=curr_test_subs)
        
        if args.subject == 'mix' and args.caption != 'style_new':
            _, test_ds = random_split(test_ds, [r_train, r_test], generator=gen)
            
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
        return None, test_loader
    else:
        raise ValueError(f"Unknown period: {period}")

if __name__ == "__main__":
    pass

if __name__ == "__main__":
    pass