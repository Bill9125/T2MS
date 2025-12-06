import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path

import openai
from dotenv import load_dotenv              # pip install python-dotenv
from tqdm import tqdm                       # pip install tqdm

# ---------- 1. 產生向量 ---------- #
def get_embedding(client, text: str,
                  model: str = "text-embedding-3-large",
                  dim: int = 128) -> list[float]:
    """呼叫 OpenAI embeddings API，回傳 128 維向量。"""
    text = text.replace("\n", " ")
    r = client.embeddings.create(input=[text], model=model, dimensions=dim)
    return r.data[0].embedding

# ---------- 2. 處理單一 clip ---------- #
def process_clip(client: openai.OpenAI, clip_dir: str, data: dict) -> None:
    subject = os.path.basename(os.path.dirname(clip_dir))
    
    category = ['correct', 'tilting_to_the_right', 'tilting_to_the_left', 'elbows_flaring', 'wrist_bending_backward', 'scapular_protraction']
    found_cats = [cat.replace('_', ' ') for cat in category if cat in subject]
    
    if not found_cats:
        classes = "unknown"
    elif len(found_cats) == 1:
        classes = found_cats[0]
    else:
        classes = ", ".join(found_cats[:-1]) + " and " + found_cats[-1]
    cap_path = path.join(clip_dir, "caption.json")

    with open(cap_path, "r", encoding="utf-8") as f:
        caption = json.load(f)

    text = caption.get("Summary", "")
    if text == "":
        print(f"Empty summary: {cap_path}")
        return
    prefix = f"The following presents the feature description for the {len(data['feature_0'])} frames of bench press. It is categorized as {classes}, with the feature sequence and described as follows: \n"
    caption["Prefix"] = prefix
    caption["Prefix_embedding"] = get_embedding(client, prefix)
    caption["Summary_embedding"] = get_embedding(client, text)

    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(caption, f, ensure_ascii=False, indent=4)

# ---------- 3. 主程式 ---------- #
def main(args) -> None:
    load_dotenv()                                   # 讀 .env
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 收集所有 clip 目錄
    subjects = glob.glob(path.join(args.caption_data_path, "*"))
    clip_dirs = [
        clip for subj in subjects
        for clip in glob.glob(path.join(subj, "*"))
    ]

    # 建立執行緒池並顯示進度 
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        for d in clip_dirs:
            subject = os.path.basename(os.path.dirname(d))
            rep = str(os.path.basename(d))
            futures.append(pool.submit(process_clip, client, d, data[subject][rep]))
        for _ in tqdm(as_completed(futures),
                    total=len(futures),
                    desc="Processing clips",
                    unit="clip"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption",
        default="Caption_explain_no_barbell_length",
        help="subject 資料夾根路徑")
    parser.add_argument(
        "--sport",
        type=str,
        choices=['benchpress', 'deadlift'],
        default='benchpress')
    args = parser.parse_args()
    args.caption_data_path = f"./Data/{args.sport}/{args.caption}"
    args.data_path = f"./Data/{args.sport}/data.json"
    main(args)
