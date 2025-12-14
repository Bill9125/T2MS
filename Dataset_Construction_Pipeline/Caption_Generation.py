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
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": f"""
                You are a senior sports data analyst specializing in correcting bench press form.
                You will receive pairwise analyses of kinematic features and a list of detected error tags for a specific repetition.

                Your reasoning must be grounded in the following **Error Classification Logic**:

                <Error_Knowledge_Base>
                    <Error_Type name="Uneven Extension (左右手伸展不均)">
                        <Manifestation>
                            <!-- 您在此填寫: 例如左手肘角度與右手肘角度的差值在上升階段超過 15 度 -->
                            [Template]: When comparing {Left_Elbow_Angle} and {Right_Elbow_Angle}, a divergence > threshold at the concentric phase indicates uneven extension.
                        </Manifestation>
                    </Error_Type>

                    <Error_Type name="Bar Path Deviation (槓鈴軌跡偏移)">
                        <Manifestation>
                            <!-- 您在此填寫: 例如肩關節角度在大重量時發生異常抖動，或垂直位移不平滑 -->
                            [Template]: Irregular fluctuations in {Shoulder_Angle} combined with non-linear {Bar_Vertical_Displacement} suggests poor bar path control.
                        </Manifestation>
                    </Error_Type>

                    <Error_Type name="Elbow Flaring (手肘外開過大)">
                        <Manifestation>
                            <!-- 您在此填寫 -->
                            [Template]: {Shoulder_Abduction_Angle} sustaining near 90 degrees throughout the descent phase.
                        </Manifestation>
                    </Error_Type>
                    
                    <!-- Add more error rules here as needed -->
                </Error_Knowledge_Base>

                Task:
                Synthesize the provided pairwise feature analyses into a comprehensive report that explains the temporal dynamics and justifies the labeled errors.
                """},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return str(completion.choices[0].message.content).strip()

def get_pairwise_completion(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": f"""
                You are an expert in biomechanics and time series analysis, specifically for resistance training exercises like the bench press.
                Your goal is to generate granular, precise, and context-aware descriptions of the relationship between two kinematic features.
                Focus on:
                1. Exact temporal synchronization (e.g., do they peak together?).
                2. Local dynamics (rising, falling, plateaus, saddle points).
                3. Biomechanical implications based on the feature definitions.
                Avoid generic summaries; strictly adhere to the data points provided.
                """},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return str(completion.choices[0].message.content).strip()

def clip_caption(features, errors):
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
    Input Data: 
    1. **Labeled Errors** for this repetition: {errors} (e.g., ['Uneven Extension', 'Bar Path Deviation'])
    2. **Pairwise Feature Analyses**:
    {combined_text}

    Task:
    1. **Holistic Reconstruction**: Summarize the overall movement flow based on the pairwise analyses.
    2. **Error Diagnosis**: specifically address the "Labeled Errors". Use the provided feature evidence (local extrema, rising/falling trends) to explain *why* and *when* these errors occurred according to the <Error_Knowledge_Base>.
    3. **Anomaly Detection**: Highlight any other notable anomalies in the data points that might contribute to these errors.
    4. **Constraint**: The output MUST be detailed (up to 2048 tokens).
    5. **Format**: Output ONLY in the following JSON format:
    Use the following JSON format:
    ```{formatted_json}```
    """
    
    start_time = time.time()
    caption = str(get_summary_completion(final_prompt)).strip()
    end_time = time.time()
    cleaned = re.sub(r"^```(json)?|```$", "", caption, flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)
    print(parsed)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    print(f"Token count: {len(encoding.encode(final_prompt))}\n")
    return parsed

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

def pairwise_summary(features):
    pair_descriptions = []
    feature_names = list(features.keys())[2:]
    
    # 使用 ThreadPoolExecutor 來管理執行緒
    with ThreadPoolExecutor(max_workers=11) as executor:
        future_to_pair = {}
        
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names[i+1:], start=i+1):
                f1_data, f2_data = features[f1], features[f2]
                f1_data = savgol_filter(f1_data, 5, 2)
                f2_data = savgol_filter(f2_data, 5, 2)
                f1_critical_points = calculate_critical_points(f1_data)
                f2_critical_points = calculate_critical_points(f2_data)
                
                # 特徵與特徵文字描述 prompt
                pair_prompt = f"""
                You are given two time series features (kinematic data from a bench press repetition) with their values, definitions, and calculated critical points:

                Feature 1: {f1}
                Definition: {feature_explaination[f1]}
                Values: {list(f1_data)}
                Critical Points (Indices & Values): {f1_critical_points}

                Feature 2: {f2}
                Definition: {feature_explaination[f2]}  
                Values: {list(f2_data)}
                Critical Points (Indices & Values): {f2_critical_points}

                Task:
                1. Analyze the detailed temporal relationship between {f1} and {f2} point-by-point.
                2. Specifically describe behaviors at **local maxima/minima**, **saddle points**, and during **rising/falling intervals**.
                3. Identify if changes in one feature precede, lag, or synchronize with the other.
                4. Relate these dynamics back to the physical movement (e.g., "as the arm extends, the angle increases rapidly until index X").
                5. The output must be comprehensive yet precise (max 512 tokens).

                Output:
                Provide a detailed narrative description focusing on the flow of movement.
                """
                
                # 提交任務到執行緒池
                future = executor.submit(get_pairwise_completion, pair_prompt)
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
        feature = savgol_filter(feature, 5, 2)
        # Min-Max normalization 到 [0,1]
        # if feature.max() != feature.min():  
        #     norm_feature = (feature - feature.min()) / (feature.max() - feature.min())
        # else:
        #     norm_feature = np.zeros_like(feature)  # 避免除以0

        plt.plot(feature, label=f'{feature_name}')  # 每條線自動不同顏色

        
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
    parser.add_argument('--dataset_name', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--max_retries', type=int, default=3)
    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name + '.yaml')
    args.data_path = f'./Data/{args.dataset_name}/data.json'
    args.output_folder = f'./Data/{args.dataset_name}/Caption_explain_no_barbell'
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    feature_list = {}
    feature_explaination = {}
    error_list = {}
    
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
                errors = [error for error in list(error_list.values()) if error in subject]
                if not path.exists(save_dir):
                    os.makedirs(save_dir)
                else:
                    print('Already exist.')
                    break
                txt_path = path.join(save_dir, 'caption.json')
                caption = clip_caption(features, errors)
                with open(txt_path, 'w', encoding="utf-8") as f:
                    json.dump(caption, f, indent=4)
                fig_path = path.join(save_dir, 'fig.jpg')
                plot_data_to_picture(features, caption, fig_path)
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