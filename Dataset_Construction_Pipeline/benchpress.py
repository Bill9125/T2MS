import os.path as path
import numpy as np
import os, glob, json, ast, csv
from scipy.signal import butter, filtfilt

class FeatureMerger:
    def __init__(self, class_dir, output_root, multi_error_path, feature):
        self.reverse_feature = {v: k for k, v in feature.items()}
        self.feature_order = list(feature.keys())
        with open(multi_error_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            me_subject = {}
            for row in reader:
                subject = row["\ufeff受試者資料夾名稱"]
                label = f'{row["錯誤1"]}_{row["錯誤2"]}'
                me_subject[subject] = label
                
        if not path.exists(output_root):
            os.makedirs(output_root)
        all_subject_feature = {}
        
        for dir in class_dir:
            subjects = glob.glob(path.join(dir, '*'))
            
            for subject in subjects:
                datasets = glob.glob(path.join(subject, '*'))
                all_angle_view_feature = {}
                
                for dataset in datasets:  # [angle, coordinate, video]
                    if path.basename(dataset) == 'angle_dataset':
                        views = glob.glob(path.join(dataset, '*'))
                        for view in views:  # [rear, top]
                            angles = glob.glob(path.join(view, '*'))
                            for angle_path in angles:
                                angle_name = path.basename(angle_path)
                                clips = glob.glob(path.join(angle_path, '*.txt'))
                                all_angle_view_feature = self.read_angle_txt(clips, all_angle_view_feature, angle_name)
                                
                    if path.basename(dataset) == 'coordinate_dataset':
                        views = glob.glob(path.join(dataset, '*'))
                        for coordinate_path in views:  # [lateral_view, rear_view, top_view]
                            clips = glob.glob(path.join(coordinate_path, '*.txt'))
                            all_angle_view_feature = self.read_coordinate_feature_txt(clips, all_angle_view_feature, view=path.basename(coordinate_path))
                
                if path.basename(subject) in me_subject:
                    all_subject_feature[f'{path.basename(subject)}_{me_subject[path.basename(subject)]}'] = all_angle_view_feature
                else:
                    all_subject_feature[f'{path.basename(subject)}_{path.basename(dir)}'] = all_angle_view_feature

        all_subject_feature = self.sort_features(all_subject_feature)
        all_subject_feature = self.filter_features(all_subject_feature)
        
        with open(path.join(output_root, 'smoothdata.json'), "w", encoding="utf-8") as f:
            json.dump(all_subject_feature, f, indent=4)
        
    def sort_features(self, d):
        if isinstance(d, dict):
            # Sort features based on the provided order if they exist in feature_order, otherwise alphabetical
            return {k: self.sort_features(v) for k, v in sorted(
                d.items(), 
                key=lambda x: (0, self.feature_order.index(x[0])) if x[0] in self.feature_order 
                else (1, x[0])
            )}
        elif isinstance(d, list):
            return [self.sort_features(i) for i in d]
        else:
            return d

    def filter_features(self, all_subject_feature):
        """
        對所有特徵進行 Butterworth 低通濾波 (1Hz)
        """
        fs = 30          # 影像 / 骨架 FPS
        cutoff = 1       # 依照 filter_test.py L20 的設定
        order = 4
        b, a = butter(order, cutoff / (fs / 2), btype='low')

        for subject, clips in all_subject_feature.items():
            for clip, features in clips.items():
                for feature_name, data in features.items():
                    # 確保數據長度足以進行濾波 (通常需要 > 3 * max(len(a), len(b)))
                    if isinstance(data, list) and len(data) > 30:
                        y = np.array(data, dtype=float)
                        # 套用雙向濾波以消除相位延遲
                        filtered_y = filtfilt(b, a, y)
                        features[feature_name] = filtered_y.tolist()
        
        return all_subject_feature

    def read_angle_txt(self, clips, all_angle_view_feature, angle_name):
        for clip_path in clips:
            with open(clip_path, 'r') as f:
                lines = f.read().strip().split('\n')
                data = ([float(line.split(',')[1]) for line in lines])
                clip = path.splitext(path.basename(clip_path))[0]
                
            if clip not in all_angle_view_feature.keys():
                all_angle_view_feature[clip] = {}
            all_angle_view_feature[clip][self.reverse_feature[angle_name]] = data
        return all_angle_view_feature

    def extract_lateral_view_feature(self, lines):
        data = np.array([[float(x) for x in line.split(',')[1:3]] for line in lines])
        return data

    def extract_rear_view_feature(self, lines):
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

    def extract_top_view_feature(self, lines):
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
        return np.array([self.process_wrist_to_shoulder_line(all_coords)])[0]

    def read_coordinate_feature_txt(self, clips, all_angle_view_feature, view):
        for clip_path in clips:
            with open(clip_path, 'r') as f:
                lines = f.read().strip().split('\n')
                clip = path.splitext(path.basename(clip_path))[0]
                
                if clip not in all_angle_view_feature.keys():
                    all_angle_view_feature[clip] = {}
                    
                if view == 'lateral_view':
                    data = self.extract_lateral_view_feature(lines) # [bar_x, bar_y]
                    all_angle_view_feature[clip][self.reverse_feature['bar_x']] = data[:,0].tolist()
                    all_angle_view_feature[clip][self.reverse_feature['bar_y']] = data[:,1].tolist()
                    
                elif view == 'rear_view':
                    data = self.extract_rear_view_feature(lines) # [left_shoulder_y, right_shoulder_y]
                    all_angle_view_feature[clip][self.reverse_feature['left_shoulder_y']] = data[:,0].tolist()
                    all_angle_view_feature[clip][self.reverse_feature['right_shoulder_y']] = data[:,1].tolist()
                    
                elif view == 'top_view':
                    data = self.extract_top_view_feature(lines) # [left_dist, right_dist]
                    all_angle_view_feature[clip][self.reverse_feature['left_dist']] = data[:,0].tolist()
                    all_angle_view_feature[clip][self.reverse_feature['right_dist']] = data[:,1].tolist()
                    
                else:
                    raise ValueError(f"Unsupported view: {view}")
                
        return all_angle_view_feature

    def perpendicular_distance(self, point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return None
        proj_len = np.dot(point_vec, line_vec) / line_len
        proj_point = line_start + (proj_len / line_len) * line_vec
        return np.linalg.norm(point - proj_point)

    def process_wrist_to_shoulder_line(self, coords_list):
        distances = []
        for pts in coords_list:
            if len(pts) == 8:
                right_wrist = pts[6]
                left_wrist = pts[7]
                right_shoulder = pts[0]
                left_shoulder = pts[1]
                line_start = np.array(right_shoulder)
                line_end = np.array(left_shoulder)
                distances.append([self.perpendicular_distance(np.array(left_wrist), line_start, line_end), self.perpendicular_distance(np.array(right_wrist), line_start, line_end)])
        return distances