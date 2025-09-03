"""
This script is part of the dataset construction pipeline for time series data.
It processes samples, generates summaries, and saves them in JSON format.
Merge all vision skeleton data into one file for caption generation.
"""
import os.path as path
import argparse, glob
from utils import merge_files, data_check

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_root', type=str)
    parser.add_argument('--multierror', type=str)
    parser.add_argument('--sport', type=str, choices=['deadlift', 'benchpress'])
    args = parser.parse_args()
    if args.sport == 'deadlift':
        class_dir = glob.glob(path.join(args.data_path, '*'))
        class_dir = [p for p in class_dir if path.isdir(p)]
        lengths = merge_files(class_dir, args.output_root, args.multierror)
        data_check(lengths, args.output_root)
    elif args.sport == 'benchpress':
        class_dir = glob.glob(path.join(args.data_path, '*'))
        class_dir = [p for p in class_dir if path.isdir(p)]
        merge_files(class_dir, args.output_root, args.multierror)
    # python ./Dataset_Construction_Pipeline/TS_data_rebuild.py --sport benchpress --data_path Data/public/BenchpressDataset --output_root Data/benchpress --multierror Data/public/BenchpressDataset/multierror.csv