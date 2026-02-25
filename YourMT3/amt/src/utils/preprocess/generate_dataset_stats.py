""" 
Generate Dataset Stats from yourmt3_indexes.

Usage: python count_events_and_files_from_dataset.py <dataset_path>

"""

import os
import sys
import glob
import json
import numpy as np
from utils.note2event import note_event2event


def generate_dataset_stats_for_all_datasets(dataset_path: os.PathLike):
    """ Count the number of notes and files in the dataset. """
    yourmt3_index_dir = os.path.join(dataset_path, "yourmt3_indexes")
    index_files = glob.glob(os.path.join(yourmt3_index_dir, '*.json'))
    output_file = os.path.join(yourmt3_index_dir, 'dataset_stats.json')
    if output_file in index_files:
        index_files.remove(output_file)  # remove output file from index files

    counts = {}
    for index_file in index_files:
        # Split-wise counts
        split_file_name = os.path.basename(index_file)

        with open(index_file, 'r') as f:
            index = json.load(f)

        file_cnt = len(index)
        event_cnt = 0
        for file_dict in index.values():
            ne_file = file_dict['note_events_file']
            note_events_dict = np.load(ne_file, allow_pickle=True).tolist()
            note_events = note_events_dict['note_events']
            events = note_event2event(note_events)
            event_cnt += len(events)

        # Update counts
        counts[split_file_name] = {'num_files': file_cnt, 'num_events': event_cnt}
        print(split_file_name, f'num_files: {file_cnt}, num_events: {event_cnt}')

    # Save counts as json
    with open(output_file, 'w') as f:
        json.dump(counts, f, indent=4)
    print(f'Saved data counts to {output_file}')


def update_dataset_stats_for_new_dataset(dataset_path: os.PathLike, split_file_name: str):
    """ Update the number of notes and files of specific dataset. """
    raise NotImplementedError


if __name__ == '__main__':
    if sys.argv == 2:
        dataset_path = sys.argv[1]
        generate_dataset_stats_for_all_datasets(dataset_path)
    else:
        print('Usage: generate_dataset_stats.py <dataset_path>')
        print('Example: python count_events_and_files_from_dataset.py ../../data')
        sys.exit(1)