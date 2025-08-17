from torch.utils.data import DataLoader, random_split
from datafactory.benchpress_dataset import Dataset_Benchpress, Datasubset
import json
import torch
import os

def loader_provider(args):
    gen = torch.Generator().manual_seed(args.general_seed)
    path = os.path.join('Data', args.dataset_path, 'data.json')
    with open(path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    full_dataset = Dataset_Benchpress(all_data)
    print(len(full_dataset))
    
    train_size = int(0.75 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = int(len(full_dataset)) - train_size - valid_size
    
    train_indices, valid_indices, test_indices = random_split(
        range(len(full_dataset)), [train_size, valid_size, test_size],
        generator=gen
    )
    
    train_dataset = Datasubset(full_dataset, train_indices, transform=False)
    valid_dataset = Datasubset(full_dataset, valid_indices, transform=False)
    test_dataset = Datasubset(full_dataset, test_indices, transform=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    pass