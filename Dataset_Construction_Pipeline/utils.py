from collections import defaultdict
import matplotlib.pyplot as plt
import os.path as path
import numpy as np
import os, re, glob, json, ast

feature = {'feature_0': 'bar_x', 'feature_1': 'bar_y', 'feature_2': 'barx/bar_y', 'feature_3': 'left_shoulder_y',
           'feature_4': 'right_shoulder_y', 'feature_5 ': 'left_dist', 'feature_6': 'right_dist', 'feature_7': 'left_elbow',
           'feature_8': 'left_shoulder', 'feature_9': 'right_elbow', 'feature_10': 'right_shoulder', 'feature_11': 'left_torso-arm',
           'feature_12': 'right_torso-arm'}
reverse_feature = {v: k for k, v in feature.items()}

def sort_features(d):
    if isinstance(d, dict):
        return {k: sort_features(v) for k, v in sorted(d.items(), key=lambda x: int(x[0].split("_")[-1]) if "feature_" in x[0] else x[0])}
    elif isinstance(d, list):
        return [sort_features(i) for i in d]
    else:
        return d

def data_check(lengths, output_root):
    plt.hist(lengths, bins=5)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Data Lengths')
    plt.savefig(path.join(output_root, 'length_distribution.png'))

def sort_files(data):
    # 儲存尾數對應的每個類別的檔案路徑
    grouped = defaultdict(dict)

    # 建立尾數到檔案路徑的映射
    for key, paths in data.items():
        for path in paths:
            filename = os.path.basename(path)
            match = re.search(r'_(\d+)\.txt$', filename)
            if match:
                suffix = match.group(1)  # 抓取尾數
                grouped[int(suffix)][key] = path  # 這邊轉成 int 方便後面排序

    # 將尾數相同的 group zip 起來（只保留同時擁有所有 keys 的）
    all_keys = set(data.keys())
    zipped = []

    for suffix in sorted(grouped.keys()):  # 按數字順序處理每個尾數
        file_dict = grouped[suffix]
        if set(file_dict.keys()) == all_keys:
            # 依據 key 排序以確保順序一致
            zipped.append(tuple(file_dict[k] for k in sorted(all_keys)))
    
    return zipped

def read_angle_txt(clips, all_angle_view_feature, angle_name):
    for clip_path in clips:
        with open(clip_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data = ([float(line.split(',')[1]) for line in lines])
            clip = path.splitext(path.basename(clip_path))[0]
            
        if clip not in all_angle_view_feature.keys():
            all_angle_view_feature[clip] = {}
        all_angle_view_feature[clip][reverse_feature[angle_name]] = data
    return all_angle_view_feature

def extract_lateral_view_feature(lines):
    data = np.array([[float(x) for x in line.split(',')[1:3]] for line in lines])
    third_col = (data[:, 0] / data[:, 1]).reshape(-1, 1)
    data = np.hstack([data, third_col])
    return data
    
def extract_rear_view_feature(lines):
    all_coords = []
    for line in lines:
        # 拆成 Frame id 和座標字串
        frame_part, coords_part = line.split(":", 1)
        frame_id = int(frame_part.replace("Frame", "").strip())
        
        # 解析成 Python list
        coords = ast.literal_eval(coords_part.strip())  # [[(x1,y1), (x2,y2), ...]]
        
        # 只取 y 值 (index=1)，並且只保留前 2 個點
        coords = [[point[1] for point in person[:2]] for person in coords]
        all_coords.extend(coords)

    # 轉 numpy array
    all_coords = np.array(all_coords, dtype=float)
    return all_coords
    
def extract_top_view_feature(lines):
    all_coords = []
    for line in lines:
        # 拆成 Frame id 和座標字串
        frame_part, coords_part = line.split(":", 1)
        frame_id = int(frame_part.replace("Frame", "").strip())
        
        # 解析成 Python list
        coords = ast.literal_eval(coords_part.strip())  # [[(x1,y1), (x2,y2), ...]]
        
        # 只取 y 值 (index=1)，並且只保留前 2 個點
        coords = [[list(point) for point in person] for person in coords]
        all_coords.extend(coords)

    # 轉 numpy array
    all_coords = np.array(all_coords, dtype=float)
    return np.array([process_wrist_to_shoulder_line(all_coords)])[0]

def read_coordinate_feature_txt(clips, all_angle_view_feature, view):
    for clip_path in clips:
        with open(clip_path, 'r') as f:
            lines = f.read().strip().split('\n')
            clip = path.splitext(path.basename(clip_path))[0]
            
            if clip not in all_angle_view_feature.keys():
                all_angle_view_feature[clip] = {}
                
            if view == 'lateral_view':
                data = extract_lateral_view_feature(lines) # [bar_x, bar_y, barx/bar_y]
                all_angle_view_feature[clip][reverse_feature['bar_x']] = data[:,0].tolist()
                all_angle_view_feature[clip][reverse_feature['bar_y']] = data[:,1].tolist()
                all_angle_view_feature[clip][reverse_feature['barx/bar_y']] = data[:,2].tolist()
                
            elif view == 'rear_view':
                data = extract_rear_view_feature(lines) # [left_shoulder_y, right_shoulder_y]
                all_angle_view_feature[clip][reverse_feature['left_shoulder_y']] = data[:,0].tolist()
                all_angle_view_feature[clip][reverse_feature['right_shoulder_y']] = data[:,1].tolist()
                
            elif view == 'top_view':
                data = extract_top_view_feature(lines) # [left_dist, right_dist]
                all_angle_view_feature[clip][reverse_feature['left_dist']] = data[:,0].tolist()
                all_angle_view_feature[clip][reverse_feature['right_dist']] = data[:,1].tolist()
                
            else:
                raise ValueError(f"Unsupported view: {view}")
            
    return all_angle_view_feature

def perpendicular_distance(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return None
    proj_len = np.dot(point_vec, line_vec) / line_len
    proj_point = line_start + (proj_len / line_len) * line_vec
    return np.linalg.norm(point - proj_point)

def process_wrist_to_shoulder_line(coords_list):
    distances = []
    for pts in coords_list:
        if len(pts) == 8:
            right_wrist = pts[6]
            left_wrist = pts[7]
            right_shoulder = pts[0]
            left_shoulder = pts[1]
            line_start = np.array(right_shoulder)
            line_end = np.array(left_shoulder)
            distances.append([perpendicular_distance(np.array(right_wrist), line_start, line_end), perpendicular_distance(np.array(left_wrist), line_start, line_end)])
    return distances


def merge_files(class_dir, output_root):
    if not path.exists(output_root):
        os.makedirs(output_root)
    all_subject_feature = {}
    
    for dir in class_dir:
        subjects = glob.glob(path.join(dir, '*'))
        
        for subject in subjects:
            datasets = glob.glob(path.join(subject, '*'))
            all_clip_feature = {}
            all_angle_view_feature = {}
            
            for dataset in datasets:  # [angle, coordinate, video]
                if path.basename(dataset) == 'angle_dataset':
                    views = glob.glob(path.join(dataset, '*'))
                    for view in views:  # [rear, top]
                        angles = glob.glob(path.join(view, '*'))
                        for angle_path in angles:
                            angle_name = path.basename(angle_path)
                            clips = glob.glob(path.join(angle_path, '*.txt'))
                            all_angle_view_feature = read_angle_txt(clips, all_angle_view_feature, angle_name)
                            
                if path.basename(dataset) == 'coordinate_dataset':
                    views = glob.glob(path.join(dataset, '*'))
                    for coordinate_path in views:  # [lateral_view, rear_view, top_view]
                        clips = glob.glob(path.join(coordinate_path, '*.txt'))
                        all_angle_view_feature = read_coordinate_feature_txt(clips, all_angle_view_feature, view=path.basename(coordinate_path))
            
            label = {'label': path.basename(dir)}
            all_clip_feature = all_angle_view_feature|label
            all_subject_feature[path.basename(subject)] = all_clip_feature
    all_subject_feature = sort_features(all_subject_feature)
    with open(path.join(output_root, 'data.json'), "w", encoding="utf-8") as f:
        json.dump(all_subject_feature, f, indent=4)
        
        
    
    #         zip_data = sort_files(info)
    #         arrange = {}
    #         for i, z in enumerate(zip_data):
    #             tmp = []
    #             for txt in z:
    #                 data = np.loadtxt(txt, delimiter=",")
    #                 data = data[:, 1:]
    #                 if path.basename(path.dirname(txt)) == 'bar':
    #                     data = data[:, 0:2]
    #                 tmp.append(data)
    #             arrange[i] = np.hstack(tmp).tolist()
    #             if len(arrange[i]) < 50 or len(arrange[i]) > 250:
    #                 print(f"Warning: Sample {i+1} in {path.basename(subject)}/{path.basename(recording)} has an unexpected length: {len(arrange[i])}")
    #                 continue
    #             lengths.append(len(arrange[i]))
    #             json.dump(arrange, open(path.join(output_dir, 'data.json'), 'w'), indent=4)
    # return lengths