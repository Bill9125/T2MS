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
from concurrent.futures import ThreadPoolExecutor, as_completed

feature_list = {'feature_0': 'bar_x', 'feature_1': 'bar_y', 'feature_2': 'barx/bar_y', 'feature_3': 'left_shoulder_y',
            'feature_4': 'right_shoulder_y', 'feature_5': 'left_dist', 'feature_6': 'right_dist', 'feature_7': 'left_elbow',
            'feature_8': 'left_shoulder', 'feature_9': 'right_elbow', 'feature_10': 'right_shoulder', 'feature_11': 'left_torso-arm',
            'feature_12': 'right_torso-arm'}

feature_explaination = {'feature_0': 'Horizontal x-axis coordinate of the barbell end in the image/frame (lateral-view).', 'feature_1': 'Vertical y-axis coordinate of the barbell end in the image/frame (lateral-view).',
                        'feature_2': 'Ratio of bar end coordinates, bar_x divided by bar_y; dimensionless.', 'feature_3': 'Vertical y-axis coordinate of the left shoulder in the rear (head-back) view.',
                        'feature_4': 'Vertical y-axis coordinate of the right shoulder in the rear (head-back) view.', 'feature_5': 'Perpendicular distance from the wrist to the extended line connecting the two shoulder joints in the top-down view.',
                        'feature_6': 'Perpendicular distance from the wrist to the extended line connecting the two shoulder joints in the top-down view.', 'feature_7': 'Elbow joint angle at the left side in the rear (head-back) view.',
                        'feature_8': 'Shoulder joint angle at the left side in the rear (head-back) view, defined by connecting line between the two shoulder joints and upper arm.', 'feature_9': 'Elbow joint angle at the right side in the rear (head-back) view.',
                        'feature_10': 'Shoulder joint angle at the right side in the rear (head-back) view, defined by connecting line between the two shoulder joints and upper arm.', 'feature_11': 'Angle at the left armpit between the torso axis and the left upper arm in the top-down view.',
                        'feature_12': 'Angle at the right armpit between the torso axis and the right upper arm in the top-down view.'}

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
    You are given multiple pairwise analyses of time series features, where each analysis describes the relationship between two features using their definitions:

    {combined_text}

    Task:
    1. Summarize these pairwise observations into **one coherent description**.
    2. Highlight the **overall temporal dynamics** and **inter-feature relationships** across the clip.
    4. Identify and retain only the **notable extreme values** (e.g., exceptionally high or low points that strongly affect dynamics).
    5. The output MUST be less than 512 tokens.
    6. DO NOT add extra explanations, markdown, or commentary.
    7. Output only in the JSON format:
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
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    print(f"Token count: {len(encoding.encode(final_prompt))}\n")
    return parsed

def pairwise_summary(features):
    pair_descriptions = []
    feature_names = list(features.keys())[3:]
    
    # 使用 ThreadPoolExecutor 來管理執行緒
    with ThreadPoolExecutor(max_workers=11) as executor:
        future_to_pair = {}
        
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names[i+1:], start=i+1):
                f1_data, f2_data = features[f1], features[f2]
                
                # 特徵與特徵文字描述 prompt
                pair_prompt = f"""
                    You are given two time series features with their values and definitions:
                    
                    {feature_list[f1]}
                    Definition: {feature_explaination[f1]}  
                    Values: {list(f1_data)}
                    Max Value: {max(f1_data)}
                    Min Value: {min(f1_data)}

                    {feature_list[f2]}  
                    Definition: {feature_explaination[f2]}  
                    Values: {list(f2_data)}  
                    Max Value: {max(f2_data)}  
                    Min Value: {min(f2_data)}

                    Task:  
                    1. Compare and analyze the temporal relationship between {f1} and {f2}.  
                    2. Highlight how their trends correlate, diverge, or interact over time, based on their definitions.  
                    3. Consider how the maximum and minimum values of both features influence their temporal dynamics.
                    4. Use a **precise and concise single sentence** (max 128 tokens).  
                    5. Focus on clarity, dynamics, and inter-feature meaning.
                """
                
                # 提交任務到執行緒池
                future = executor.submit(get_completion, pair_prompt)
                future_to_pair[future] = (f1, f2)
        
        # 收集結果
        for future in as_completed(future_to_pair):
            f1, f2 = future_to_pair[future]
            try:
                desc = future.result()
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
    parser.add_argument('--data_path', type=str, default='./Data/benchpress/data.json')
    parser.add_argument('--output_path', type=str, default='./Data/benchpress/Caption_explain_no_barbell')
    parser.add_argument('--max_retries', type=int, default=3)
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    
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
                    save_dir = path.join(args.output_path, subject, clip)
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