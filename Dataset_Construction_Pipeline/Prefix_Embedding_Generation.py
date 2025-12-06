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

def main(args):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
    
    # 儲存向量
    os.makedirs(args.caption_data_path, exist_ok=True)
    with open(f"{args.caption_data_path}/embeddings.json", "w") as f:
        json.dump(embeddings, f)

if __name__ == "__main__":
    load_dotenv()      # 讀取 .env 檔
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sport",
        type=str,
        choices=['benchpress', 'deadlift'],
        default='benchpress')
    parser.add_argument(
        "--text",
        type=str,
        default="")
    args = parser.parse_args()
    args.output_path = f"./Data/{args.sport}"
    main(args)