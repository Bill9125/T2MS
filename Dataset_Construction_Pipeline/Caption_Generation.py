import argparse
import json
import random
import openai
import time
import tiktoken
from matplotlib import pyplot as plt
import yaml
import os
import re
import os.path as path
from dotenv import load_dotenv
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

plot_lock = threading.Lock()

def get_summary_completion(user_prompt, system_prompt):
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
    )
    return str(completion.choices[0].message.content).strip()

def clip_caption(features, errors, prompt_template, formatted_json_str, system_prompt, style_name):
    # 直接使用原始特徵數據，不經過 single_feature_description
    feature_descs = []
    for f_name, f_data in features.items():
        formatted_data = [f"{value:.3f}" for value in f_data]
        feature_descs.append(f"Feature {f_name}: " + ", ".join(formatted_data))
    
    combined_text = "\n".join(feature_descs)
    
    indicators_list = []
    if isinstance(errors, dict):
        for defect_name, conditions in errors.items():
            if conditions:
                indicators_list.extend(conditions)
    
    # 針對舊版 explain 風格的捷徑：如果沒有錯誤，直接回傳預設正確姿勢。
    # 但對於新版 style_new，我們依然要讓 LLM 分析並提取特徵數值！
    if not indicators_list and style_name == 'explain':
        parsed = {
            "Indicator_Evaluations": [],
            "Summary": "Standard and perfect correct posture without any mistakes."
        }
        return [parsed], feature_descs
    
    indicators_text = "    None"
    if indicators_list:
        indicators_text = "\n".join([f"    {idx+1}. {cond}" for idx, cond in enumerate(indicators_list)])

    # 動態填入 Template
    final_prompt = prompt_template.format(
        combined_text=combined_text,
        indicators_text=indicators_text,
        formatted_json=formatted_json_str
    )
    parseds = []
    for i in range(1):
        start_time = time.time()
        caption = None
        for attempt in range(3):
            try:
                caption = str(get_summary_completion(final_prompt, system_prompt)).strip()
                break
            except Exception as e:
                print(f"API Error on attempt {attempt+1}: {e}")
                time.sleep(2)
        
        if not caption:
            continue
            
        end_time = time.time()
        cleaned = re.sub(r"^```(json)?|```$", "", caption, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            continue
            
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        encoding = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = len(encoding.encode(final_prompt))
        output_tokens = len(encoding.encode(caption))
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}\n")
        parseds.append(parsed)
    return parseds, feature_descs

def plot_data_to_picture(features, save_path, feature_list):
    with plot_lock:
        plt.figure(figsize=(12, 8))
        for feature_name, feature in features.items():
            feature = np.array(feature, dtype=float)
            plt.plot(feature, label=f'{feature_name}')  # 每條線自動不同顏色

        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend(fontsize=8)
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', '-d', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--prompt_style', type=str, default='explain', help='Prompt style defined in yaml config')
    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name + '.yaml')
    args.data_path = f'./Data/{args.dataset_name}/data.json'
    args.output_folder = f'./Data/{args.dataset_name}/Caption_{args.prompt_style}'
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    feature_list = {}
    feature_explaination = {}
    error_list = {}
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if 'prompt_styles' not in config or args.prompt_style not in config['prompt_styles']:
        raise ValueError(f"Prompt style '{args.prompt_style}' not found in {args.config}")
    
    sys_prompt = config['prompt_styles'][args.prompt_style]['system_prompt']
    prompt_template = config['prompt_styles'][args.prompt_style]['prompt_template']
    formatted_json_str = config['prompt_styles'][args.prompt_style]['formatted_json']
    
    def process_single_clip(subject, clip, features, error_list, feature_list, args):
        retries = 0
        while retries < args.max_retries:
            try:
                print(f'current sample is {subject} on {clip}')
                save_dir = path.join(args.output_folder, subject, clip)
                if not path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                if "correct" in subject.lower():
                    errors = {}
                else:
                    errors = {error: error_list[error] for error in error_list if error in subject}
                
                json_path = path.join(save_dir, 'caption.json')

                # 檢查這筆資料是否已經有完整的 caption.json
                if path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding="utf-8") as f:
                            data = json.load(f)
                            # 如果發現裡面已經有 Summary_0，這筆資料就沒有遺漏，直接 return 跳過
                            if "Summary_0" in data:
                                print(f"Skipping {subject} on {clip}: Summary already exists.")
                                return
                    except Exception:
                        # 檔案可能是空的或是壞掉的，那就往下繼續重新生成
                        pass
                        
                summary, feature_descs = clip_caption(features, errors, prompt_template, formatted_json_str, sys_prompt, args.prompt_style)

                local_caption = {}
                for i, parsed in enumerate(summary):
                    if 'Indicator_Evaluations' in parsed:
                        local_caption[f'Indicator_Evaluations_{i}'] = parsed['Indicator_Evaluations']
                    if 'Feature_Highlights' in parsed:
                        local_caption[f'Feature_Highlights_{i}'] = parsed['Feature_Highlights']
                    if 'Summary' in parsed:
                        local_caption[f'Summary_{i}'] = parsed['Summary']
                
                for desc in feature_descs:
                    if ": " in desc:
                        k, v = desc.split(": ", 1)
                        local_caption[k] = v
                
                with open(json_path, 'w', encoding="utf-8") as f:
                    json.dump(local_caption, f, indent=4)
                
                fig_path = path.join(save_dir, 'fig.jpg')
                plot_data_to_picture(features, fig_path, feature_list)
                print(f'current sample is {subject} on {clip} finished')
                return 

            except Exception as e:
                print(f"Error occurred in {subject}/{clip}: {e}. Retrying {retries + 1}/{args.max_retries}...")
                time.sleep(2)
                retries += 1
        
        if retries == args.max_retries:
            error_message = f"Failed to process sample {subject} on {clip} after {args.max_retries} retries."
            with open('error_log.txt', 'a') as file:
                file.write(error_message + "\n")
    
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
    
    # error = ['Barbell_moving_away_from_the_shins', 'Hips_rising_before_the_barbell_leaves_the_ground', 'Barbell_colliding_with_the_knees', 'Lower_back_rounding']
    count = 0
    for subject, clips in data.items():
        # if error[count] not in subject:
        #     continue
        # print(subject)
        # count += 1
        # Prepare work items
        work_items = []
        all_clips = list(clips.keys())
        # 如果是 style_new，每個 subject 隨機挑選 3 個 clip
        if args.prompt_style == 'style_new' and len(all_clips) > 3:
            selected_clips = random.sample(all_clips, 3)
        else:
            selected_clips = all_clips

        for clip in selected_clips:
            features = clips[clip]
            if 'body_length' in features:
                del features['body_length']
            work_items.append((subject, clip, features))
            
        # Use ThreadPoolExecutor to process clips in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_single_clip, sub, clp, feat, error_list, feature_list, args) 
                for sub, clp, feat in work_items
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Unhandled error in thread: {e}")
