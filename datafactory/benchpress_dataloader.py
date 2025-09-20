from torch.utils.data import Dataset, DataLoader, random_split
from datafactory.benchpress_dataset import BenchpressT2SDataset
import torch
import numpy as np

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
            batch_texts, batch_xs, batch_embeddings = zip(*data_list)
            batch_texts = [torch.from_numpy(text) if isinstance(text, np.ndarray) else text for text in batch_texts]
            batch_xs = torch.stack(
                [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in batch_xs]
            )
            batch_embeddings = torch.stack(
                [torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding for embedding in batch_embeddings]
            )
            batches.append((batch_texts, batch_xs, batch_embeddings))
    return batches

def loader_provider(args):
    dataset1 = BenchpressT2SDataset(
        json_path=args.dataset_path,
        caption_root = args.caption_data_path,
        emb_dim=args.embedding_dim,
        data_dim=args.split_base_num
    )
    dataset2 = BenchpressT2SDataset(
        json_path=args.dataset_path,
        caption_root = args.caption_data_path,
        emb_dim=args.embedding_dim,
        data_dim=args.split_base_num*2
    )
    dataset3 = BenchpressT2SDataset(
        json_path=args.dataset_path,
        caption_root = args.caption_data_path,
        emb_dim=args.embedding_dim,
        data_dim=args.split_base_num*4
    )
    dataset = AlternatingDataset(dataset1, dataset2, dataset3)
    
    # 可重現的隨機切分
    gen = torch.Generator().manual_seed(args.general_seed)
    r_train, r_valid, r_test = (0.85, 0.05, 0.1)
    assert abs(r_train + r_valid + r_test - 1.0) < 1e-8, "split_ratio must sum to 1.0"

    train_ds, valid_ds, test_ds = random_split(dataset, [r_train, r_valid, r_test], generator=gen)

    common = dict(batch_size=args.batch_size, collate_fn=custom_collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common) # type: ignore
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **common) # type: ignore
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common) # type: ignore
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    pass