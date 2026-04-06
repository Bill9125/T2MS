from collections import defaultdict
import os.path as path
import re, glob, json
import pandas as pd

class FeatureMerger:
    def __init__(self, class_dir, output_root, multierror_path, feature):
        self.reverse_feature = {v: k for k, v in feature.items()}
        with open(multierror_path, encoding="utf-8") as f:
            me_subject = json.load(f)
        rename_list, pass_list = self._make_list(me_subject)
        data = self._collect_data(class_dir, rename_list, pass_list)
        self._save_to_JSON(data, output_root)
        
        
    def _make_list(self, me_subject):
        pass_list = defaultdict(list) # key: subject_set_error , value: list of clip numbers to skip
        rename_list = defaultdict(list) # key: subject_set_*error , value: [new_name, list of clip numbers]
        for subject, multis in me_subject.items():
            for multi in multis:
                for i, error in enumerate(multi):
                    err = error['error']
                    set = error['set']
                    clips = error['clips']
                    if i == 0:
                        rename = f'{subject}_{set}_{err}_' + '_'.join([e['error'] for j, e in enumerate(multi) if j != i])
                        rename_list[f'{subject}_{set}_{err}'] = [rename ,clips]
                        print(f'Rename {subject}_{set}_{err} to {rename} : {clips}')
                    else:
                        pass_list[f'{subject}_{set}_{err}'] = clips
        return rename_list, pass_list
    
    def _collect_data(self, class_dir, rename_list, pass_list):
        data = defaultdict(dict)
        for dir in class_dir:
            subjects = glob.glob(path.join(dir, '*'))
            for subject in subjects:
                sets = glob.glob(path.join(subject, '*'))
                for set in sets:
                    datasets = glob.glob(path.join(set, '*'))
                    key = f'{path.basename(subject)}_{path.basename(set)}_{path.basename(dir)}'
                    if path.join(set, 'Chessboard') not in datasets: # skip the set without 3D data
                        # print(f'No 3D set: {set}')
                        continue
                    for dataset in datasets:  # [Angle, Chessboard, Coordinate, Clips]
                        if path.basename(dataset) == 'Angle':
                            angles = glob.glob(path.join(dataset, '3D', '*.csv'))
                            if key in pass_list:
                                pass_clips = pass_list[key]
                                angles = [angle for angle in angles if self._extract_clip_number(angle) not in pass_clips]
                            if angles:
                                all_clip_angle_features = self._read_angle_csv(angles)
                                target_key = rename_list[key][0] if key in rename_list else key
                                self._merge_clip_features(data[target_key], all_clip_angle_features)
                        if path.basename(dataset) == 'Coordinate':
                            coordinates = glob.glob(path.join(dataset, 'Bar', '*.csv'))
                            if key in pass_list:
                                pass_clips = pass_list[key]
                                coordinates = [coordinate for coordinate in coordinates if self._extract_clip_number(coordinate) not in pass_clips]
                            if coordinates:
                                all_clip_coordinate_features = self._read_coordinate_csv(coordinates)
                                target_key = rename_list[key][0] if key in rename_list else key
                                self._merge_clip_features(data[target_key], all_clip_coordinate_features)
        return data
    
    def _merge_clip_features(self, target_dict, new_features):
        '''Merge new features into the target dictionary, keyed by clip number.'''
        for clip_num, features in new_features.items():
            if clip_num not in target_dict:
                target_dict[clip_num] = features
            else:
                target_dict[clip_num].update(features)

    def _extract_clip_number(self, filename):
        '''Extract clip number from filename using regex.'''
        filename = path.basename(filename)
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group(0))
        else:
            return None
        
    def _read_angle_csv(self, angles):
        all_clip_angle_features = {}
        # Angle CSV format (3D): frame_index, left_knee, left_hip, right_knee, right_hip, body_length, left_torso_angle, right_torso_angle
        angle_col_names = ["frame_index", "left_knee", "left_hip", "right_knee", "right_hip", "body_length", "left_torso_angle", "right_torso_angle"]
        for angle_path in angles:
            angle_features = {}
            # print(f'Reading angle {angle_path}')
            df = pd.read_csv(
                angle_path,
                header=None,
                names=angle_col_names,
                index_col=0,
                dtype={c: "float64" for c in angle_col_names[1:]},
            )
            angle_features = df.to_dict(orient="list")
            angle_features['body_length'] = [100*v for v in angle_features['body_length']]
            all_clip_angle_features[self._extract_clip_number(angle_path)] = angle_features
        return all_clip_angle_features
    
    def _read_coordinate_csv(self, coordinates):
        all_clip_coordinate_features = {}
        # Bar CSV format: frame_index, bar_x, bar_y, bar_w, bar_h
        bar_col_names = ["frame_index", "bar_x", "bar_y", "bar_w", "bar_h"]
        for coordinate_path in coordinates:
            coordinate_features = {}
            # print(f'Reading coordinate {coordinate_path}')
            df = pd.read_csv(
                coordinate_path,
                header=None,
                names=bar_col_names,
                index_col=0,
                dtype={c: "float64" for c in bar_col_names[1:]},
            )
            # Only keep bar_x and bar_y as requested
            df = df[["bar_x", "bar_y"]]
            coordinate_features = df.to_dict(orient="list")
            all_clip_coordinate_features[self._extract_clip_number(coordinate_path)] = coordinate_features
        return all_clip_coordinate_features

    def _save_to_JSON(self, data, output_root):
        output_path = path.join(output_root, 'data.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)