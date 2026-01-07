import argparse
import os
# import openai
# from dotenv import load_dotenv
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

def plot_filter_comparison(features, base_save_path, feature_list):
    # Set style for a more premium look
    plt.style.use('ggplot') 
    
    filter_names = ['Original', 'Savgol', 'Twopass_Savgol', 'Butterworth_1Hz', 'Butterworth_2Hz', 'Butterworth_3Hz']
    titles = [
        'Original Feature Data', 
        'Savitzky-Golay Filtered', 
        'Two-pass Savitzky-Golay Filtered',
        'Butterworth Filtered (1Hz)',
        'Butterworth Filtered (2Hz)',
        'Butterworth Filtered (3Hz)'
    ]
    norm_options = [True, False]
    
    # Use a professional color palette
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(features)))
    
    for is_norm in norm_options:
        norm_suffix = "normalized" if is_norm else "raw"
        
        for idx, (filter_name, base_title) in enumerate(zip(filter_names, titles)):
            plt.figure(figsize=(12, 6), dpi=100)
            title = f"{base_title} ({norm_suffix})"
            
            for i, (feature_key, feature_data) in enumerate(features.items()):
                feature_name_label = feature_list.get(feature_key, feature_key)
                
                # Skip bar-related coordinates as requested
                if feature_name_label in ['barx/bar_y']:
                    continue
                
                data = np.array(feature_data, dtype=float)
                color = colors[i]
                
                # Apply filters
                if filter_name == 'Savgol':
                    plot_data = savgol_filter(data, window_length=9, polyorder=2)
                elif filter_name == 'Twopass_Savgol':
                    window_size = 5 
                    f_pass = savgol_filter(data, window_length=window_size, polyorder=2)
                    b_pass = savgol_filter(f_pass[::-1], window_length=window_size, polyorder=2)
                    plot_data = b_pass[::-1]
                elif 'Butterworth' in filter_name:
                    fs = 30          # 影像 / 骨架 FPS
                    order = 4
                    # 從名字提取 cutoff: Butterworth_1Hz -> 1
                    try:
                        cutoff = float(filter_name.split('_')[1].replace('Hz', ''))
                    except:
                        cutoff = 2 # fallback
                    
                    b, a = butter(order, cutoff / (fs / 2), btype='low')
                    plot_data = filtfilt(b, a, data)
                else: # Original
                    plot_data = data
                
                # Apply normalization if requested
                if is_norm:
                    if plot_data.max() != plot_data.min():
                        plot_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
                    else:
                        plot_data = np.zeros_like(plot_data)
                    
                plt.plot(plot_data, color=color, label=feature_name_label, linewidth=1.5, alpha=0.9)
                
            plt.title(title, fontsize=16, fontweight='bold', pad=15)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylabel('Normalized Value' if is_norm else 'Value', fontsize=12)
            plt.xlabel("Frame Index", fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            
            # Construct filename: e.g., filter_comparison_butterworth_1hz_normalized.png
            save_path = base_save_path.replace('.png', f'_{filter_name.lower()}_{norm_suffix}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.abspath(save_path)}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--max_retries', type=int, default=3)
    args = parser.parse_args()
    args.config = os.path.join('.', 'config', args.dataset_name + '.yaml')
    args.data_path = f'./Data/{args.dataset_name}/data.json'
    args.output_folder = f'./Data/{args.dataset_name}/Caption_explain_no_barbell'
    # load_dotenv()
    # api_key = os.getenv("OPENAI_API_KEY")
    # client = openai.OpenAI(api_key=api_key)
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
            fig_path = './Dataset_Construction_Pipeline/filter_comparison.png'
            plot_filter_comparison(features, fig_path, feature_list)
            break
        break
