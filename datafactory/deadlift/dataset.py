from torch.utils.data import Dataset
import torch
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import os.path as path
import os

class DeadliftT2SDataset(Dataset):
    """
    輸出：((text, x, embedding), dataset_idx)
      - text: 預設為 "subject/clip" 字串，可透過 text_builder 自訂
      - x: Tensor 形狀 [n_f, T]，n_f=特徵數，T=時間長度（可用 max_length 做裁剪/補零以固定）
      - embedding: Tensor [E]，預設為 zeros(E)
    """
    def __init__(
        self,
        feat_name: list,
        json_path: str,
        caption_root: str,
        period: str,
        emb_dim: int = 128,
        data_dim: int = 48,
        allowed_subjects: list = None,
    ):
        category = ['correct', 'Barbell_colliding_with_the_knees', 'Barbell_moving_away_from_the_shins', 'Hips_rising_before_the_barbell_leaves_the_ground', 'Lower_back_rounding']
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        self.records = []  # list of ((text, x[n_f,T], embedding[E]), dataset_idx)
        lengths = []
        max_len = 0
        min_len = float('inf')  # 初始化為無限大
        
        for subject, clips in all_data.items():
            # 受試者獨立過濾邏輯
            if allowed_subjects is not None and subject not in allowed_subjects:
                continue
                
            for clip, feat_dict in clips.items():
                caption_path = path.join(caption_root, subject, clip, 'caption.json')
                if not os.path.exists(caption_path):
                    text = ""
                    embedding = np.zeros(emb_dim, dtype=np.float32)
                else:
                    with open(caption_path, 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        text = data.get('Summary_0', "")
                        embedding = data.get('embedding', np.zeros(emb_dim, dtype=np.float32))
                    
                # 收集同一個 clip 內所有特徵為 1D 時序 [T]
                seqs_T = []
                T_list = []
                for k in feat_name:
                    x = torch.as_tensor(feat_dict[k], dtype=torch.float32)  # [T] 或 [T, D_f]
                    if x.dim() == 1:
                        seqs_T.append(x)
                        T_list.append(x.size(0))
                    else:
                        raise ValueError(f"Feature '{k}' must be [T], got {tuple(x.size())}")

                # 同 clip 內時間長度一致性檢查
                if len(set(T_list)) != 1:
                    detail = ", ".join([f"{kk}:{ll}" for kk, ll in zip(feat_name, T_list)])
                    print(f"Inconsistent time length (subject={subject}, clip={clip}): {detail}")
                    continue
                
                # 更新最小和最大長度
                current_T = T_list[0]  # 同一個 clip 內所有特徵長度相同
                lengths.append(current_T)
                max_len = max(max_len, current_T)
                min_len = min(min_len, current_T)

                # [n_f, T]
                x_nfT = torch.stack(seqs_T, dim=0)
                Tcur = x_nfT.size(1)
                
                if Tcur<10:
                    print(f'{subject}', f'{clip}', f'{Tcur}')
                    continue
                
                # 對齊到目標長度 (data_dim)
                target_T = 0
                if period == 'train':
                    target_T = self._map_target_len(Tcur, data_dim)
                elif data_dim > 0:
                    target_T = data_dim

                if target_T > 0:
                    Ttar = target_T
                    x_1cT = x_nfT.unsqueeze(0)  # [1, n_f, T]
                    if Tcur > Ttar:
                        x_1cT = F.adaptive_avg_pool1d(x_1cT, output_size=Ttar)
                    elif Tcur < Ttar:
                        x_1cT = F.interpolate(x_1cT, size=Ttar, mode='linear', align_corners=True)
                    x_nfT = x_1cT.squeeze(0)
                elif period == 'train':
                    continue

                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding)
                elif not torch.is_tensor(embedding):
                    embedding = torch.as_tensor(embedding, dtype=torch.float32)

                self.records.append((text, x_nfT, embedding, subject, clip))
    
    def _map_target_len(self, T: int, target_T):
        if target_T == 48:
            if T < 81:
                return target_T
            else:
                return 0
        elif target_T == 96:
            if 81 <= T < 99:
                return target_T
            else:
                return 0
        elif target_T == 192:
            if T >= 99:
                return target_T
            else:
                return 0
        else:
            raise ValueError(f'Undefined length {target_T}.')
        
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]  # (text, x[n_f,T], embedding[E])
    
    