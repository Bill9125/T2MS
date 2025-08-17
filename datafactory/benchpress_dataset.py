from torch.utils.data import Dataset
import torch
import random

class Dataset_Benchpress(Dataset):
    def __init__(self, all_data):
        self.features = []
        self.labels = []
        for subject, data in all_data.items():
            for label, features in data.items():
                ground_truth = list(map(int, label.split("_")))
                for feature in features.values():
                    self.features.append(torch.tensor(feature).float())
                    self.labels.append(torch.tensor(ground_truth).float())
        self.features = torch.stack(self.features)
        self.labels = torch.stack(self.labels)
        self.dim = self.features.shape[-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

class Datasubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.time_stretch(x, random.uniform(0.8, 1.2))
            x = self.add_gaussian_noise(x, std=0.01)
        return x, y

    def time_stretch(self, x, stretch_factor):
        # 假設 x.shape = (T, F)
        T, F = x.shape
        new_T = int(T * stretch_factor)
        x_stretched = torch.nn.functional.interpolate(
            x.unsqueeze(0).permute(0, 2, 1),  # (1, F, T)
            size=new_T,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1).squeeze(0)  # 回到 (T, F)
        if new_T < T:
            pad = torch.zeros(T - new_T, F, dtype=x.dtype, device=x.device)
            x_stretched = torch.cat([x_stretched, pad], dim=0)
        else:
            x_stretched = x_stretched[:T]
        return x_stretched

    def add_gaussian_noise(self, x, std=0.01):
        noise = torch.randn_like(x) * std
        return x + noise
    
    