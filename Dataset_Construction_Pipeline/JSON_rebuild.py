"""
This script is part of the dataset construction pipeline for time series data.
It processes samples, generates summaries, and saves them in JSON format.
Merge all vision skeleton data into one file for caption generation.
"""
import os.path as path
import argparse, glob
import yaml

def get_feature_cfg(args):
    feature_list = {}
    feature_explaination = {}
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f = config[args.sport]['feature']
        for id, [name, defn] in f.items():
            feature_list[f'feature_{id}'] = name['name']
            feature_explaination[f'feature_{id}'] = defn['definition']
    return feature_list, feature_explaination

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_root', type=str)
    parser.add_argument('--sport', type=str, choices=['deadlift', 'benchpress'])
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    feature, _ = get_feature_cfg(args)
            
    if args.sport == 'deadlift':
        from deadlift import FeatureMerger
        class_dir = glob.glob(path.join(args.data_path, '*'))
        class_dir = [p for p in class_dir if path.isdir(p)]
        args.multierror = path.join(args.data_path, 'multierror.json')
        FeatureMerger(class_dir, args.output_root, args.multierror, feature)
        
    elif args.sport == 'benchpress':
        from benchpress import FeatureMerger
        class_dir = glob.glob(path.join(args.data_path, '*'))
        class_dir = [p for p in class_dir if path.isdir(p)]
        args.multierror = path.join(args.data_path, 'multierror.csv')
        FeatureMerger(class_dir, args.output_root, args.multierror, feature)
    # python ./Dataset_Construction_Pipeline/TS_data_rebuild.py --sport benchpress --data_path Data/public/BenchpressDataset --output_root Data/benchpress --multierror Data/public/BenchpressDataset/multierror.csv