import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
def plot_loss_curve(loss_list, save_path, filename='loss_curve.png'):
    if len(loss_list) == 0:
        print("loss_list is empty, skipping plotting.")
        return
    plt.figure(figsize=(10, 6))
    x = [i for i in range(len(loss_list))]
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"Loss curve saved to {full_path}")
    
def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True