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
def process_clip(client: openai.OpenAI, clip_dir: str) -> None:
    cap_path = path.join(clip_dir, "caption.json")

    with open(cap_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("Summary", "")
    data["embedding"] = get_embedding(client, text)

    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ---------- 3. 主程式 ---------- #
def main(caption_data_path: str) -> None:
    load_dotenv()                                   # 讀 .env
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 收集所有 clip 目錄
    subjects = glob.glob(path.join(caption_data_path, "*"))
    clip_dirs = [
        clip for subj in subjects
        for clip in glob.glob(path.join(subj, "*"))
    ]

    # 建立執行緒池並顯示進度
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(process_clip, client, d) for d in clip_dirs]
        for _ in tqdm(as_completed(futures),
                      total=len(futures),
                      desc="Processing clips",
                      unit="clip"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption_data_path",
        default="./Data/deadlift/Caption_with_feature_explaination",
        help="subject 資料夾根路徑")
    args = parser.parse_args()
    main(args.caption_data_path)
