# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import os
import json
import torch
import numpy as np
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, Sampler
from torch.utils.data import RandomSampler, WeightedRandomSampler
from operator import itemgetter
from typing import List, Tuple, Union, Iterator, Optional
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from config.config import shared_cfg


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`. From catalyst library.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows to use any sampler in distributed mode.
    From https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def discount_to_target(samples: np.ndarray, target_sum: int) -> np.ndarray:
    """Discounts samples to target sum.

    NOTE: this function is deprecated.

    This function adjusts an array of sample values so that their sum equals a target sum, while ensuring 
    that each element remains greater than or equal to 1 and attempting to maintain a distribution similar 
    to the original.

    Example 1:
    samples = np.array([3, 1, 1, 1, 1, 1])
    target_sum = 7
    discounted_samples = discount_to_target(samples, target_sum)
    # [2, 1, 1, 1, 1, 1]

    Example 2:
    samples = np.array([3,1, 10, 1, 1, 1])
    target_sum = 7
    # [1, 1, 2, 1, 1, 1]

    Parameters:
    samples (np.ndarray): Original array of sample values.
    target_sum (int): The desired sum of the sample array.

    Returns:
    np.ndarray: Adjusted array of sample values whose sum should equal the target sum, 
    and where each element is greater than or equal to 1.
    """
    samples = samples.copy().astype(int)
    if samples.sum() <= target_sum:
        samples[0] += 1
        return samples

    while samples.sum() > target_sum:
        # indices of all elements larger than 1
        indices_to_discount = np.where(samples > 1)[0]
        if indices_to_discount.size == 0:
            # No elements left to discount, we cannot reach target_sum without going below 1
            print("Cannot reach target sum without going below 1 for some elements.")
            return samples
        discount_count = int(min(len(indices_to_discount), samples.sum() - target_sum))
        indices_to_discount = indices_to_discount[:discount_count]
        samples[indices_to_discount] -= 1
    return samples


def create_merged_train_dataset_info(data_preset_multi: dict, data_home: Optional[os.PathLike] = None):
    """Create merged dataset info from data preset multi.   
    Args:
        data_preset_multi (dict): data preset multi
        data_home (os.PathLike, optional): path to data home. If None, used the path defined 
            in config/config.py. 
    
    Returns:
        dict: merged dataset info
    """
    train_dataset_info = {
        "n_datasets": 0,
        "n_notes_per_dataset": None,  # TODO: not implemented yet...
        "n_files_per_dataset": [],
        "dataset_names": [],  # dataset names by order of merging file lists
        "data_split_names": [],  # dataset names by order of merging file lists
        "index_ranges": [],  # index ranges of each dataset in the merged file list
        "dataset_weights": None,  # pre-defined list of dataset weights for sampling, if available
        "merged_file_list": {},
    }

    if data_home is None:
        data_home = shared_cfg["PATH"]["data_home"]
    assert os.path.exists(data_home)

    for dp in data_preset_multi["presets"]:
        train_dataset_info["n_datasets"] += 1

        dataset_name = data_preset_single_cfg[dp]["dataset_name"]
        train_dataset_info["dataset_names"].append(dataset_name)
        train_dataset_info["data_split_names"].append(dp)

        # load file list for train split
        if isinstance(data_preset_single_cfg[dp]["train_split"], str):
            train_split_name = data_preset_single_cfg[dp]["train_split"]
            file_list_path = os.path.join(data_home, 'yourmt3_indexes',
                                          f'{dataset_name}_{train_split_name}_file_list.json')
            # check if file list exists
            if not os.path.exists(file_list_path):
                raise ValueError(f"File list {file_list_path} does not exist.")
            _file_list = json.load(open(file_list_path, 'r'))
        elif isinstance(data_preset_single_cfg[dp]["train_split"], dict):
            _file_list = data_preset_single_cfg[dp]["train_split"]
        else:
            raise ValueError("Invalid train split.")

        # merge file list
        start_idx = len(train_dataset_info["merged_file_list"])
        for i, v in enumerate(_file_list.values()):
            train_dataset_info["merged_file_list"][start_idx + i] = v
        train_dataset_info["n_files_per_dataset"].append(len(_file_list))
        train_dataset_info["index_ranges"].append((start_idx, start_idx + len(_file_list)))

    # set dataset weights
    if "weights" in data_preset_multi.keys() and data_preset_multi["weights"] is not None:
        train_dataset_info["dataset_weights"] = data_preset_multi["weights"]
        assert len(train_dataset_info["dataset_weights"]) == train_dataset_info["n_datasets"]
    else:
        train_dataset_info["dataset_weights"] = np.ones(train_dataset_info["n_datasets"])
        print("No dataset weights specified, using equal weights for all datasets.")
    return train_dataset_info


def get_random_sampler(dataset, num_samples):
    if torch.distributed.is_initialized():
        return DistributedSamplerWrapper(sampler=RandomSampler(dataset, num_samples=num_samples))
    else:
        return RandomSampler(dataset, num_samples=num_samples)


def get_weighted_random_sampler(dataset_weights: List[float],
                                dataset_index_ranges: List[Tuple[int]],
                                num_samples_per_epoch: Optional[int] = None,
                                replacement: bool = True) -> torch.utils.data.sampler.Sampler:
    """Get distributed weighted random sampler.
    Args:
        dataset_weights (List[float]): list of dataset weights of n length for n_datasets
        dataset_index_ranges (List[Tuple[int]]): list of dataset index ranges
        n_samples_per_epoch (Optional[int]): number of samples per epoch, typically length of 
            entire dataset. Defaults to None. If None, the total number of samples is calculated.
        replacement (bool, optional): replacement. Defaults to True.
    Returns:
        (distributed) weighted random sampler
    """
    assert len(dataset_weights) == len(dataset_index_ranges)

    sample_weights = []
    n_total_samples_in_datasets = dataset_index_ranges[-1][1]
    if len(dataset_weights) > 1 and len(dataset_index_ranges) > 1:
        for dataset_weight, index_range in zip(dataset_weights, dataset_index_ranges):
            assert dataset_weight >= 0
            n_samples_in_dataset = index_range[1] - index_range[0]
            sample_weight = dataset_weight * (1 - n_samples_in_dataset / n_total_samples_in_datasets)
            # repeat the same weight for the number of samples in the dataset
            sample_weights += [sample_weight] * (index_range[1] - index_range[0])
    elif len(dataset_weights) == 1 and len(dataset_index_ranges) == 1:
        # Single dataset
        sample_weights = [1] * n_total_samples_in_datasets

    if num_samples_per_epoch is None:
        num_samples_per_epoch = n_total_samples_in_datasets

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples_per_epoch, replacement=replacement)

    if torch.distributed.is_initialized():
        return DistributedSamplerWrapper(sampler=sampler)
    else:
        return sampler


def get_list_of_weighted_random_samplers(num_samplers: int,
                                         dataset_weights: List[float],
                                         dataset_index_ranges: List[Tuple[int]],
                                         num_samples_per_epoch: Optional[int] = None,
                                         replacement: bool = True) -> List[torch.utils.data.sampler.Sampler]:
    """Get list of distributed weighted random samplers.
    Args:
        dataset_weights (List[float]): list of dataset weights of n length for n_datasets
        dataset_index_ranges (List[Tuple[int]]): list of dataset index ranges
        n_samples_per_epoch (Optional[int]): number of samples per epoch, typically length of 
            entire dataset. Defaults to None. If None, the total number of samples is calculated.
        replacement (bool, optional): replacement. Defaults to True.
    
    Returns:
        List[(distributed) weighted random sampler]
    """
    assert num_samplers > 0
    samplers = []
    for i in range(num_samplers):
        samplers.append(
            get_weighted_random_sampler(dataset_weights, dataset_index_ranges, num_samples_per_epoch, replacement))
    return samplers
