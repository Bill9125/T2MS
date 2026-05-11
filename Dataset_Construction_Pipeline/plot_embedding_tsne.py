import argparse
import glob
import json
import os
from os import path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_prompt_tsne(args):
    dataset_path = f"./Data/{args.dataset_name}"
    
    # 搜尋該資料集下所有以 Caption_ 開頭的資料夾 (代表不同的 prompt style)
    caption_folders = glob.glob(path.join(dataset_path, "Caption_*"))
    
    if not caption_folders:
        print(f"No caption folders found in {dataset_path}")
        return

    all_embeddings = []
    all_labels = []
    all_styles = []

    for folder in caption_folders:
        style_name = path.basename(folder).replace("Caption_", "")
        subjects = glob.glob(path.join(folder, "*"))
        
        for subj in subjects:
            subj_name = path.basename(subj)
            # 判斷錯誤類別 (簡單拆解)
            if "correct" in subj_name.lower():
                label = "Correct"
            else:
                # 假設資料夾名稱格式為 subject_xxx_exp_errorname
                parts = subj_name.split("_")
                label = "_".join(parts[3:]) if len(parts) > 3 else "Error"

            clips = glob.glob(path.join(subj, "*"))
            for clip in clips:
                cap_path = path.join(clip, "caption.json")
                if path.exists(cap_path):
                    with open(cap_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        emb = data.get("embedding")
                        if emb and isinstance(emb, list) and len(emb) == 128:
                            all_embeddings.append(emb)
                            all_labels.append(label)
                            all_styles.append(style_name)

    if not all_embeddings:
        print("No valid embeddings found across the folders.")
        return

    # 將資料轉為 NumPy 陣列
    X = np.array(all_embeddings)
    
    # 使用 PCA 先初步降維，再使用 t-SNE (增強穩定性)
    print(f"Performing t-SNE on {len(X)} embeddings...")
    pca = PCA(n_components=min(50, len(X)))
    X_pca = pca.fit_transform(X)
    
    # Perplexity 不能大於樣本數
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    # 開始繪圖
    plt.figure(figsize=(12, 10))
    
    unique_labels = list(set(all_labels))
    unique_styles = list(set(all_styles))
    
    # 定義顏色對應錯誤類別，形狀對應 Prompt Style
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    markers = ['o', 'X', 's', '^', 'D', 'v', '<', '>']
    
    for i, label in enumerate(unique_labels):
        for j, style in enumerate(unique_styles):
            idx = [k for k in range(len(all_labels)) if all_labels[k] == label and all_styles[k] == style]
            if idx:
                plt.scatter(
                    X_tsne[idx, 0], 
                    X_tsne[idx, 1], 
                    color=colors[i], 
                    marker=markers[j % len(markers)], 
                    label=f"{label} ({style})",
                    alpha=0.7,
                    s=60
                )

    plt.title(f"t-SNE Projection of Text Embeddings across Prompt Styles ({args.dataset_name})", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # 將 legend 移至圖片外側
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    save_path = f"./results/tsne_prompt_comparison_{args.dataset_name}.png"
    os.makedirs("./results", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', '-d', type=str, choices=['benchpress', 'deadlift'], default='benchpress')
    args = parser.parse_args()
    plot_prompt_tsne(args)
