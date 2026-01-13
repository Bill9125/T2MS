import argparse
import json
import openai
import time
import tiktoken
from matplotlib import pyplot as plt
import os.path as path
import os
import re
import textwrap
from dotenv import load_dotenv
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_completion(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": 
                "You're an expert in multi-feature time series summarization. Generate precise, concise, and context-aware descriptions that reflect the dynamics and relationships among multiple variables. Focus on clarity and informativeness. Avoid unnecessary text or generic explanations."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return str(completion.choices[0].message.content).strip()

def clip_caption(features):
    # 先拿到每對 feature 的描述
    pairwise_descs = pairwise_summary(features)
    combined_text = "\n".join(pairwise_descs)
    formatted_json = """
        {
            "Summary": "..."
        }
    """

    # 做總結
    final_prompt = f"""
    You are given multiple pairwise analyses of time series features, where each analysis describes the relationship between two features:

    {combined_text}

    Task:
    1. Summarize these pairwise observations into **one coherent description**.  
    2. The summary should highlight the **overall temporal dynamics** and **inter-feature relationships** across the clip with data points.  
    3. The output MUST be less than 1024 tokens.  
    4. DO NOT add extra explanations, markdown, or commentary.  
    5. Output only in the JSON format:
    Use the following JSON format:
    ```{formatted_json}```
    """
    
    start_time = time.time()
    caption = str(get_completion(final_prompt)).strip()
    end_time = time.time()
    cleaned = re.sub(r"^```(json)?|```$", "", caption, flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)
    print(parsed)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    encoding = tiktoken.encoding_for_model("gpt-4o")
    print(f"Token count: {len(encoding.encode(final_prompt))}\n")
    return parsed

def pairwise_summary(features):
    pair_descriptions = []
    feature_names = list(features.keys())
    
    # 使用 ThreadPoolExecutor 來管理執行緒
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pair = {}
        
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names[i+1:], start=i+1):
                f1_data, f2_data = features[f1], features[f2]
                
                # 特徵與特徵文字描述 prompt
                pair_prompt = f"""
                You are given two time series features with their values and definitions:

                Name: {feature_list[f1]}  
                Definition: {feature_explaination[f1]}  
                Values: {list(f1_data)}

                Name: {feature_list[f2]}  
                Definition: {feature_explaination[f2]}  
                Values: {list(f2_data)}

                Task:  
                1. Compare and analyze the temporal relationship between {feature_list[f1]} and {feature_list[f2]}.  
                2. Highlight how their trends correlate, diverge, or interact over time, based on their definitions.  
                3. Use a **precise and concise single sentence** (max 128 tokens).  
                4. Focus on clarity, dynamics, and inter-feature meaning, and raw numbers.
                """
                
                # 提交任務到執行緒池
                future = executor.submit(get_completion, pair_prompt)
                future_to_pair[future] = (f1, f2)
        
        # 收集結果
        for future in as_completed(future_to_pair):
            f1, f2 = future_to_pair[future]
            try:
                desc = future.result()
                print(desc)
                pair_descriptions.append(desc)
            except Exception as exc:
                print(f'Pair {f1}-{f2} generated an exception: {exc}')
    
    return pair_descriptions

def plot_data_to_picture(features, text, save_path):
    plt.figure(figsize=(12, 8))
    for feature_name, feature in features.items():
        feature = np.array(feature, dtype=float)
        # Min-Max normalization 到 [0,1]
        if feature.max() != feature.min():  
            norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        else:
            norm_feature = np.zeros_like(feature)  # 避免除以0

        plt.plot(norm_feature, label=f'{feature_name}:{feature_list[feature_name]}, min :{feature.min():.4f}, max :{feature.max():.4f}')  # 每條線自動不同顏色

        
    # 自動換行：每 60 字符換一行
    wrapped_text = textwrap.fill(text['Summary'], width=75)
    plt.title(wrapped_text, fontsize=10, loc='center')
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend(fontsize=8)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='benchpress')
    parser.add_argument('--data_path', type=str, default='./Data/Benchpress/smoothdata.json')
    parser.add_argument('--output_path', type=str, default='./Data/Benchpress/Caption_explain_no_barbell/Caped_data.json')
    parser.add_argument('--max_retries', type=int, default=3)
    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name + '.yaml')
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    
    feature_list = {}
    feature_explaination = {}
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f_cfg = config['features']
        for id, [name, defn] in f_cfg.items():
            feat_name = name['name']
            feature_list[feat_name] = feat_name
            feature_explaination[feat_name] = defn['definition']
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
        
    for subject, clips in data.items():
        for clip, features in clips.items():
            string = ""
            retries = 0
            while retries < args.max_retries:
                try:
                    print(f'current sample is {subject} on {clip}')
                    caption = clip_caption(features)
                    save_dir = path.join(path.dirname(args.output_path), subject, clip)
                    print(save_dir)
                    if not path.exists(save_dir):
                        os.makedirs(save_dir)
                    txt_path = path.join(save_dir, 'caption.json')
                    with open(txt_path, 'w', encoding="utf-8") as f:
                        json.dump(caption, f, indent=4)
                    fig_path = path.join(save_dir, 'fig.jpg')
                    plot_data_to_picture(features, caption, fig_path)
                    break
                
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying {retries + 1}/{args.max_retries}...")
                    retries += 1
                
                if retries == args.max_retries:
                    error_message = f"Failed to process sample {subject} on {clip} after {args.max_retries} retries."
                    with open(f'error_log.txt', 'a') as file:
                        file.write(error_message + "\n")