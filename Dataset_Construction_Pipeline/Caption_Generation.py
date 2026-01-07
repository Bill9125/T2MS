import argparse
import json
import openai
import time
import tiktoken
from matplotlib import pyplot as plt
import yaml
import os
import re
import textwrap
import os.path as path
from dotenv import load_dotenv
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import savgol_filter

def get_summary_completion(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system",
             "content": f"""
                You're an expert in multi-feature time series summarization. Generate precise, concise, and context-aware descriptions that reflect the dynamics and relationships among multiple variables. Focus on clarity and informativeness. Avoid unnecessary text or generic explanations.
                """},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
    )
    return str(completion.choices[0].message.content).strip()

def get_single_feature_completion(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": f"""
                You are an expert in time series analysis. Your goal is to generate granular and precise descriptions.
                Avoid generic summaries; strictly adhere to the data points provided.
                """},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return str(completion.choices[0].message.content).strip()

def clip_caption(features, errors, feature_list):
    # 先拿到每對 feature 的描述
    feature_descs = single_feature_description(features, feature_list)
    combined_text = "\n".join(feature_descs)
    formatted_json = """
        {
            "Summary": "..."
        }
    """
    # 做總結
    final_prompt = f"""
    You are given a set of feature time-series descriptions and a confirmed classification with possible diagnostic indicators.
    Task:
    1. Analyze the provided feature time-series descriptions.
    2. Identify and extract which diagnostic indicators are present in the features.
    3. NOT to verify if the classification is correct.
    4. The output MUST be consistent with actual feature analyses.
    5. The output MUST be less than 1024 tokens.
    Given the classification: tilting_to_the_right
    Possible Diagnostic Indicators:
    1. When 'bar_y' increases 'right_elbow' decreases faster than 'left_elbow' in descent speed.
    2. When 'bar_y' decreases 'left_elbow' rises faster than 'right_elbow' in ascent speed.
    3. Throughout the beginning and end stages of the movement, 'left_elbow' is larger than 'right_elbow'.
    Given each single feature analysis
    ```{combined_text}```
    Use the following JSON format:
    ```{formatted_json}```
    """
    print(final_prompt)
    parseds = []
    for i in range(5):
        start_time = time.time()
        caption = str(get_summary_completion(final_prompt)).strip()
        end_time = time.time()
        cleaned = re.sub(r"^```(json)?|```$", "", caption, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        encoding = tiktoken.encoding_for_model("gpt-5-mini")
        print(f"Token count: {len(encoding.encode(final_prompt))}\n")
        parseds.append(parsed)
    return parseds, feature_descs

def calculate_critical_points(sequence):
    try:
        data = np.array(sequence, dtype=float)
    except:
        return "None"
        
    if len(data) < 3:
        return "None"
        
    points = []
    
    # Calculate gradients
    grad1 = np.gradient(data)
    grad2 = np.gradient(grad1)
    
    critical_indices = set()
    
    # Find local maxima and minima
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            points.append((i, "Top Position", data[i]))
            critical_indices.add(i)
        elif data[i] < data[i-1] and data[i] < data[i+1]:
            points.append((i, "Bottom Position", data[i]))
            critical_indices.add(i)
            
    # Find inflection points (Zero crossing of 2nd derivative)
    for i in range(1, len(data) - 1):
        if i in critical_indices:
            continue
            
        # Check for sign change in 2nd derivative suggesting inflection
        if grad2[i-1] * grad2[i+1] < 0:
            points.append((i, "Inflection Point", data[i]))
            
    # Sort points by index
    points.sort(key=lambda x: x[0])
    
    # Format the output
    result = [f"Index {p[0]} ({p[1]}): {p[2]:.2f}" for p in points]
            
    return ", ".join(result) if result else "None"

def single_feature_description(features, feature_list):
    feature_names = list(features.keys())
    
    # 使用 ThreadPoolExecutor 來管理執行緒
    with ThreadPoolExecutor(max_workers=11) as executor:
        future_to_pair = {}
        
        for i, f in enumerate(feature_names):
            f_data = features[f]
            f_data = savgol_filter(f_data, 5, 2)
            # f_critical_points = calculate_critical_points(f_data)
            formatted_data = [f"{i + 1}, {value:.3f}" for i, value in enumerate(f_data)]
            formatted_string = "\n".join(formatted_data)
            
            # 特徵與特徵文字描述 prompt
            feature_prompt = f"""
            You are given a time series feature with its name and values:
            Feature name: {feature_list[f]}
            Task:
            1.Summarize the observed trend in the given time series data.
            2.The output MUST be less than 256 tokens.
            3.The output description MUST be consistent with the actual trend characteristics of the time series.
            Given the time series data
            ```{formatted_string}```
            """
            
            # 提交任務到執行緒池
            future = executor.submit(get_single_feature_completion, feature_prompt)
            future_to_pair[future] = (f)
        
        # 收集結果
        descriptions = []
        for future in as_completed(future_to_pair):
            f = future_to_pair[future]
            try:
                desc = future.result()
                descriptions.append(desc)
            except Exception as exc:
                print(f'Feature {f} generated an exception: {exc}')
    return descriptions

def plot_data_to_picture(features, save_path, feature_list):
    plt.figure(figsize=(12, 8))
    for feature_name, feature in features.items():
        feature = np.array(feature, dtype=float)
        feature = savgol_filter(feature, 5, 2)
        # # Min-Max normalization 到 [0,1]
        # if feature.max() != feature.min():  
        #     norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        # else:
        #     norm_feature = np.zeros_like(feature)  # 避免除以0

        plt.plot(feature, label=f'{feature_list[feature_name]}')  # 每條線自動不同顏色

    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend(fontsize=8)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--max_retries', type=int, default=3)
    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name + '.yaml')
    args.data_path = f'./Data/{args.dataset_name}/smoothdata.json'
    args.output_folder = f'./Data/{args.dataset_name}/Caption_explain_no_barbell'
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    feature_list = {}
    feature_explaination = {}
    error_list = {}
    caption = {}
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f = config['features']
        for id, [name, defn] in f.items():
            feature_list[f'feature_{id}'] = name['name']
            feature_explaination[f'feature_{id}'] = defn['definition']
        f = config['mistakes']
        for id, [name, defn] in f.items():
            error_list[name['name']] = defn['definition']
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
        
    for subject, clips in data.items():
        for clip, features in clips.items():
            string = ""
            retries = 0
            while retries < args.max_retries:
                # try:
                print(f'current sample is {subject} on {clip}')
                save_dir = path.join(args.output_folder, subject, clip)
                errors = [error for error in list(error_list.keys()) if error in subject]
                txt_path = path.join(save_dir, 'caption.json')
                summary, feature_descs = clip_caption(features, errors, feature_list)
                for idx, desc in enumerate(feature_descs):
                    caption[feature_list[f'feature_{idx}']] = desc
                for i, parsed in enumerate(summary):
                    caption[f'Summary_{i}'] = parsed['Summary']
                if not path.exists(save_dir):
                    os.makedirs(save_dir)
                else:
                    print('Already exist.')
                    break
                with open(txt_path, 'w', encoding="utf-8") as f:
                    json.dump(caption, f, indent=4)
                fig_path = path.join(save_dir, 'fig.jpg')
                plot_data_to_picture(features, fig_path, feature_list)
                break  # 成功後跳出重試迴圈 
                
                # except Exception as e:
                #     print(f"Error occurred: {e}. Retrying {retries + 1}/{args.max_retries}...")
                #     retries += 1
                
            if retries == args.max_retries:
                error_message = f"Failed to process sample {subject} on {clip} after {args.max_retries} retries."
                with open(f'error_log.txt', 'a') as file:
                    file.write(error_message + "\n")
            break
        break
