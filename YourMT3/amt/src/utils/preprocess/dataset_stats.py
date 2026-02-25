# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""dataset_stats.py"""
import os
import json
import glob
import numpy as np
from typing import Optional, List

STAT_FILE_NAME = "dataset_stats.json"


def generate_dataset_stats(data_home: os.PathLike, dataset_name: Optional[str] = None) -> None:
    """Generate dataset stats for a given dataset.

    Args:
        data_home: Path to the data directory.
        dataset_name: Name of the dataset to (re)generate stats for. If None, generate MISSING stats for all
          datasets.
    """
    stat_file = os.path.join(data_home, 'yourmt3_indexes', STAT_FILE_NAME)
    if os.path.exists(stat_file):
        print(f"Loading existing dataset stats file: {stat_file}")
        with open(stat_file, 'r') as f:
            stats = json.load(f).items()
    else:
        print(f"Creating new dataset stats file: {stat_file}")
        stats = {}

    # Collect all existing yourmt3 indexes
    indexes = glob.glob(os.path.join(data_home, 'yourmt3_indexes', '*_file_list.json'))
    for index_file in indexes:
        dataset_name = os.path.basename(index_file).split('_')[0]
        split_name = os.path.basename(index_file).split('_')[1]
