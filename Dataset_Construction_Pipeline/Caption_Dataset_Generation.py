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
    return completion.choices[0].message.content

def one_sample_data_summary(data):
    formatted_json = """
        {
            "Trend Analysis": "..."
        }
        """
    user_prompt = f"""
        You are given a segment of multi-feature time series data.
        Your task:
        1. Analyze and summarize the temporal trends and the relationships among features.
        2. ONLY output your result in the exact JSON format shown below.
        3. The value of "Trend Analysis" MUST be a single descriptive string that integrates all features into one coherent narrative.
        4. DO NOT output per-feature dictionaries, nested structures, or numeric key-value pairs.
        5. The output MUST be less than 512 tokens.
        6. The description MUST be consistent with the actual trends present in the data.
        7. DO NOT add extra explanations, markdown, or commentary.
        Given the multi-feature time series data
        ```{data}```​
        Use the following JSON format:
        ```{formatted_json}```
        """

    start_time = time.time()
    one_sample_completion = get_completion(user_prompt)
    end_time = time.time()
    raw_output = one_sample_completion.strip()
    # 移除 ```json 或 ``` 包裹
    cleaned = re.sub(r"^```(json)?|```$", "", raw_output, flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)
    print(parsed)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    encoding = tiktoken.encoding_for_model("gpt-4o")
    print(f"Token count: {len(encoding.encode(user_prompt))}")
    return parsed

def plot_data_to_picture(features, text, save_path):
    for feature_name, feature in features.items():
        plt.plot(feature, label=feature_name)  # 每條線自動不同顏色
        
    # 自動換行：每 60 字符換一行
    wrapped_text = textwrap.fill(text['Trend Analysis'], width=60)
    plt.title(wrapped_text, fontsize=10, loc='center')
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument('--data_path', type=str, default='./Data/benchpress/data.json')
    paser.add_argument('--output_path', type=str, default='./Data/benchpress/Caped_data.json')
    paser.add_argument('--max_retries', type=int, default=3)
    args = paser.parse_args()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    for subject, clips in data.items():
        for clip, features in clips.items():
            if clip == 'label':
                continue
            string = ""
            retries = 0
            for feature_name, feature in features.items(): # data in every clip
                string += f"{feature_name} : {list(feature)}\n"
            
            while retries < args.max_retries:
                try:
                    print(f'current sample is {subject} on {clip}')
                    one_sample_text = one_sample_data_summary(string)
                    save_dir = path.join(path.dirname(args.output_path), clips['label'], subject, clip)
                    print(save_dir)
                    if not path.exists(save_dir):
                        os.makedirs(save_dir)
                    txt_path = path.join(save_dir, 'caption.json')
                    with open(txt_path, 'w', encoding="utf-8") as f:
                        json.dump(one_sample_text, f, indent=4)
                    fig_path = path.join(save_dir, 'fig.jpg')
                    plot_data_to_picture(features, one_sample_text, fig_path)
                    break
                
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying {retries + 1}/{args.max_retries}...")
                    retries += 1
                
            if retries == args.max_retries:
                error_message = f"Failed to process sample {subject} on {clip} after {args.max_retries} retries."
                with open(f'error_log.txt', 'a') as file:
                    file.write(error_message + "\n")
                    