import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def plot_distribution(dataset_name):
    data_path = f'./Data/{dataset_name}/data.json'
    if not os.path.exists(data_path):
        print(f"Error: Path {data_path} does not exist.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    plot_data = {}
    
    if dataset_name == 'benchpress':
        features_to_plot = ['left_dist', 'right_dist']
        colors = {'left_dist': '#3498db', 'right_dist': '#e74c3c'}
        feat_labels = {'left_dist': 'Left Dist', 'right_dist': 'Right Dist'}
        groups = ['All']
        group_colors = {'All': '#3498db'}
    elif dataset_name == 'deadlift':
        features_to_plot = ['bar_x']
        groups = ['Away from Shins', 'Normal']
        group_colors = {'Away from Shins': '#e74c3c', 'Normal': '#2ecc71'}
        feat_labels = {'bar_x': 'Bar X'}
    else:
        print(f"Unknown dataset: {dataset_name}")
        return

    for g in groups:
        plot_data[g] = {f: [] for f in features_to_plot}

    for subject, clips in data.items():
        if dataset_name == 'deadlift':
            group = 'Away from Shins' if 'Barbell_moving_away_from_the_shins' in subject else 'Normal'
        else:
            group = 'All'
            
        for clip, features in clips.items():
            for feat in features_to_plot:
                if feat in features:
                    vals = np.array(features[feat])
                    if len(vals) > 0:
                        vals = vals - np.min(vals)
                        plot_data[group][feat].extend(vals.tolist())
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    has_data = False
    for g in groups:
        for feat in features_to_plot:
            vals = np.array(plot_data[g][feat])
            if len(vals) > 0:
                if dataset_name == 'deadlift':
                    label = f"{g} (Mean: {np.mean(vals):.2f})"
                    color = group_colors[g]
                else:
                    label = f"{feat_labels[feat]} (Mean: {np.mean(vals):.2f})"
                    color = colors[feat]
                
                sns.kdeplot(vals, fill=True, label=label, color=color, alpha=0.5)
                has_data = True

    if not has_data:
        print("No data found for requested features and filters.")
        return

    title = f'Feature KDE Distribution: {dataset_name.capitalize()} (Relative to Clip Min)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Displacement (Value - Clip Min)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    output_path = f'distribution_{dataset_name}.png'
    plt.savefig(output_path, dpi=300)
    print(f"Distribution plot saved to {output_path}")

def plot_enhanced_trend(dataset_name, remove_outliers=True):
    data_path = f'./Data/{dataset_name}/data.json'
    if not os.path.exists(data_path): 
        print(f"Error: Path {data_path} does not exist.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    
    if dataset_name == 'benchpress':
        status_fn = lambda subj: "Elbow Flaring" if "elbows_flaring" in subj else "Normal"
        palette = {"Normal": "#2ecc71", "Elbow Flaring": "#e74c3c"}
        hue_order = ["Normal", "Elbow Flaring"]
        row_facet = "Side"
    elif dataset_name == 'deadlift':
        status_fn = lambda subj: "Away from Shins" if "Barbell_moving_away_from_the_shins" in subj else "Normal"
        palette = {"Normal": "#2ecc71", "Away from Shins": "#e74c3c"}
        hue_order = ["Normal", "Away from Shins"]
        row_facet = None
    else:
        return

    for subject_id, clips in data.items():
        status = status_fn(subject_id)
        if status is None: continue
        
        for clip_id, features in clips.items():
            valid_parts = []
            if dataset_name == 'benchpress':
                if 'left_dist' in features and 'right_dist' in features:
                    valid_parts = [('Left', features['left_dist']), ('Right', features['right_dist'])]
            elif dataset_name == 'deadlift':
                if 'bar_x' in features:
                    valid_parts = [('Bar', features['bar_x'])]
            
            for side, vals in valid_parts:
                vals = np.array(vals)
                n_frames = len(vals)
                if n_frames < 2: continue
                # Subtract min of each time series
                vals = vals - np.min(vals)
                for i in range(n_frames):
                    time_pct = (i / (n_frames - 1)) * 100
                    records.append({
                        "Time (%)": time_pct,
                        "Value": vals[i],
                        "Side": side,
                        "Status": status
                    })

    df = pd.DataFrame(records)
    if df.empty:
        print("Empty dataset.")
        return

    if remove_outliers:
        print(f"Filtering outliers for {dataset_name} (>95% CI)...")
        df['Time_Bin'] = (df['Time (%)'] // 5) * 5
        
        filtered_subsets = []
        for (side, status, time_bin), group in df.groupby(['Side', 'Status', 'Time_Bin']):
            should_filter = False
            if dataset_name == 'benchpress' and status == "Elbow Flaring":
                should_filter = True
            elif dataset_name == 'deadlift' and status == "Away from Shins":
                should_filter = True
            
            if should_filter:
                lower = group['Value'].quantile(0.025)
                upper = group['Value'].quantile(0.975)
                group = group[(group['Value'] >= lower) & (group['Value'] <= upper)]
            filtered_subsets.append(group)
        
        if not filtered_subsets:
            print("No groups found for filtering.")
            return
            
        df = pd.concat(filtered_subsets).reset_index(drop=True)
        print(f"Filtering complete. Remaining records: {len(df)}")

    sns.set_theme(style="whitegrid")
    
    g = sns.FacetGrid(df, row=row_facet, hue="Status", palette=palette, 
                      height=5, aspect=2, sharey=True)
    
    g.map(sns.scatterplot, "Time (%)", "Value", alpha=0.3, s=5, edgecolor=None, 
          hue_order=hue_order)
    
    g.add_legend(title="Movement Status")
    g.set_axis_labels("Movement Progress (%)", "Displacement (Value - Clip Min)")
    
    plt.subplots_adjust(top=0.9)
    title_suffix = " (Outliers Removed, Scatter View)" if remove_outliers else " (Scatter View)"
    main_title = f'Relative Feature Distribution over Time: {dataset_name.capitalize()}'
    g.fig.suptitle(f'{main_title}{title_suffix}', fontsize=16, fontweight='bold')

    output_path = f'enhanced_trend_scatter_{dataset_name}.png'
    plt.savefig(output_path, dpi=300)
    print(f"Scatter trend plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze feature distribution.')
    parser.add_argument('--dataset_name', type=str, default='benchpress', help='Name of the dataset directory')
    parser.add_argument('--mode', type=str, default='all', choices=['dist', 'trend', 'all'], help='Plotting mode')
    parser.add_argument('--filter', action='store_true', help='Remove outliers for specific group')
    args = parser.parse_args()
    
    if args.mode in ['dist', 'all']:
        plot_distribution(args.dataset_name)
    if args.mode in ['trend', 'all']:
        plot_enhanced_trend(args.dataset_name, remove_outliers=args.filter or True)

        