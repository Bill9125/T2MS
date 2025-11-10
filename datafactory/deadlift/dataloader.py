from torch.utils.data import Dataset, DataLoader, random_split
from .dataset import DeadliftT2SDataset
import torch
import numpy as np
import os

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
    grouped_data = {0: [], 1: [], 2: []}
    grouped_data = {
        idx: [data for data, dataset_idx in batch if dataset_idx == idx]
        for idx in grouped_data.keys()
    }
    batches = []
    for idx, data_list in grouped_data.items():
        if data_list:
            batch_texts, batch_xs, batch_embeddings, subjects = zip(*data_list)
            batch_texts = [torch.from_numpy(text) if isinstance(text, np.ndarray) else text for text in batch_texts]
            batch_subjects = [torch.from_numpy(subject) if isinstance(subject, np.ndarray) else subject for subject in subjects]
            batch_xs = torch.stack(
                [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in batch_xs]
            )
            batch_embeddings = torch.stack(
                [torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding for embedding in batch_embeddings]
            )
            batches.append((batch_texts, batch_xs, batch_embeddings, batch_subjects))
    return batches

def loader_provider(args, period='train'):
    if period == 'train':
        dataset1 = DeadliftT2SDataset(
            json_path=os.path.join(args.dataset_root, args.dataset_name, 'data.json'),
            caption_root = os.path.join(args.dataset_root, args.dataset_name, args.caption),
            emb_dim=args.embedding_dim,
            data_dim=args.split_base_num,
            period=period
        )
        dataset2 = DeadliftT2SDataset(
            json_path=os.path.join(args.dataset_root, args.dataset_name, 'data.json'),
            caption_root = os.path.join(args.dataset_root, args.dataset_name, args.caption),
            emb_dim=args.embedding_dim,
            data_dim=args.split_base_num*2,
            period=period
        )
        dataset3 = DeadliftT2SDataset(
            json_path=os.path.join(args.dataset_root, args.dataset_name, 'data.json'),
            caption_root = os.path.join(args.dataset_root, args.dataset_name, args.caption),
            emb_dim=args.embedding_dim,
            data_dim=args.split_base_num*4,
            period=period
        )
        dataset = AlternatingDataset(dataset1, dataset2, dataset3)
        common = dict(batch_size=args.batch_size, collate_fn=custom_collate_fn)
    
    elif period == 'test':
        dataset = DeadliftT2SDataset(
            json_path=os.path.join(args.dataset_root, args.dataset_name, 'data.json'),
            caption_root = os.path.join(args.dataset_root, args.dataset_name, args.caption),
            emb_dim=args.embedding_dim,
            data_dim=0,
            period=period
        )
        common = dict(batch_size=args.batch_size)
    else:
        raise ValueError(f"Not expected period")
    
    # 可重現的隨機切分
    gen = torch.Generator().manual_seed(args.general_seed)
    r_train, r_test = (0.9, 0.1)
    assert abs(r_train + r_test - 1.0) < 1e-8, "split_ratio must sum to 1.0"

    train_ds, test_ds = random_split(dataset, [r_train, r_test], generator=gen)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common) # type: ignore
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common) # type: ignore
    return train_loader, test_loader

if __name__ == "__main__":
    pass