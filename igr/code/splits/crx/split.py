import os
import sys
import argparse
import json
import random

code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, code_dir)

import utils.general as utils

def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True, type=str, help='Database path')
    parser.add_argument('--add-symmetric-ids', action='store_true', help='Add symmetric identifiers')
    parser.add_argument('--training-set-ratio', type=float, default=0.8, help='Training set ratio')

    args = parser.parse_args()

    os.path.dirname(os.path.realpath(__file__))
    with open(args.dataset_path) as f:
        samples = json.load(f)['database']['samples']


    # Shuffle input samples
    random.shuffle(samples)

    # Create splits
    n_train_samples = int(len(samples)*args.training_set_ratio)
    all_samples_ids = [s['case_identifier'] for s in samples]
    train_samples_ids = [i for i in all_samples_ids[:n_train_samples]]
    test_samples_ids = [i for i in all_samples_ids[n_train_samples:]]

    # Add symmetrics to each split
    if args.add_symmetric_ids:
        all_samples_ids += [f'{i}_sym' for i in all_samples_ids]
        train_samples_ids += [f'{i}_sym' for i in train_samples_ids]
        test_samples_ids += [f'{i}_sym' for i in test_samples_ids]

    # Save splits
    crx_split_dir = os.path.dirname(os.path.realpath(__file__))
    write_json(all_samples_ids, os.path.join(crx_split_dir, 'all.json'))
    write_json(train_samples_ids, os.path.join(crx_split_dir, 'train.json'))
    write_json(test_samples_ids, os.path.join(crx_split_dir, 'test.json'))