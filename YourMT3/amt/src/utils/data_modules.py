# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
""" data_modules.py """
from typing import Optional, Dict, List, Any
import os
import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from utils.datasets_train import get_cache_data_loader
from utils.datasets_eval import get_eval_dataloader
from utils.datasets_helper import create_merged_train_dataset_info, get_list_of_weighted_random_samplers
from utils.task_manager import TaskManager
from config.config import shared_cfg
from config.config import audio_cfg as default_audio_cfg
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg


class AMTDataModule(LightningDataModule):

    def __init__(
            self,
            data_home: Optional[os.PathLike] = None,
            data_preset_multi: Dict[str, Any] = {
                "presets": ["musicnet_mt3_synth_only"],
            },  # only allowing multi_preset_cfg. single_preset_cfg should be converted to multi_preset_cfg
            task_manager: TaskManager = TaskManager(task_name="mt3_full_plus"),
            train_num_samples_per_epoch: Optional[int] = None,
            train_random_amp_range: List[float] = [0.6, 1.2],
            train_stem_iaug_prob: Optional[float] = 0.7,
            train_stem_xaug_policy: Optional[Dict] = {
                "max_k": 3,
                "tau": 0.3,
                "alpha": 1.0,
                "max_subunit_stems": 12,  # the number of subunit stems to be reduced to this number of stems
                "p_include_singing":
                    0.8,  # probability of including singing for cross augmented examples. if None, use base probaility.
                "no_instr_overlap": True,
                "no_drum_overlap": True,
                "uhat_intra_stem_augment": True,
            },
            train_pitch_shift_range: Optional[List[int]] = None,
            audio_cfg: Optional[Dict] = None) -> None:
        super().__init__()

        # check path existence
        if data_home is None:
            data_home = shared_cfg["PATH"]["data_home"]
        if os.path.exists(data_home):
            self.data_home = data_home
        else:
            raise ValueError(f"Invalid data_home: {data_home}")
        self.preset_multi = data_preset_multi
        self.preset_singles = []
        # e.g. [{"dataset_name": ..., "train_split": ..., "validation_split":...,}, {...}]
        for dp in self.preset_multi["presets"]:
            if dp not in data_preset_single_cfg.keys():
                raise ValueError("Invalid data_preset")
            self.preset_singles.append(data_preset_single_cfg[dp])

        # task manager
        self.task_manager = task_manager

        # train num samples per epoch, passed to the sampler
        self.train_num_samples_per_epoch = train_num_samples_per_epoch
        assert shared_cfg["BSZ"]["train_local"] % shared_cfg["BSZ"]["train_sub"] == 0
        self.num_train_samplers = shared_cfg["BSZ"]["train_local"] // shared_cfg["BSZ"]["train_sub"]

        # train augmentation parameters
        self.train_random_amp_range = train_random_amp_range
        self.train_stem_iaug_prob = train_stem_iaug_prob
        self.train_stem_xaug_policy = train_stem_xaug_policy
        self.train_pitch_shift_range = train_pitch_shift_range

        # train data info
        self.train_data_info = None  # to be set in setup()

        # validation/test max num of files
        self.val_max_num_files = data_preset_multi.get("val_max_num_files", None)
        self.test_max_num_files = data_preset_multi.get("test_max_num_files", None)

        # audio config
        self.audio_cfg = audio_cfg if audio_cfg is not None else default_audio_cfg

    def set_merged_train_data_info(self) -> None:
        """Collect train datasets and create info...

        self.train_dataset_info = {
            "n_datasets": 0,
            "n_notes_per_dataset": [],
            "n_files_per_dataset": [],
            "dataset_names": [],  # dataset names by order of merging file lists
            "train_split_names": [],  # train split names by order of merging file lists
            "index_ranges": [],  # index ranges of each dataset in the merged file list
            "dataset_weights": [],  # pre-defined list of dataset weights for sampling, if available
            "merged_file_list": {},
        }
        """
        self.train_data_info = create_merged_train_dataset_info(self.preset_multi)
        print(
            f"AMTDataModule: Added {len(self.train_data_info['merged_file_list'])} files from {self.train_data_info['n_datasets']} datasets to the training set."
        )

    def setup(self, stage: str):
        """
        Prepare data args for the dataloaders to be used on each stage.
        `stage` is automatically passed by pytorch lightning Trainer.
        """
        if stage == "fit":
            # Set up train data info
            self.set_merged_train_data_info()

            # Distributed Weighted random sampler for training
            actual_train_num_samples_per_epoch = self.train_num_samples_per_epoch // shared_cfg["BSZ"][
                "train_local"] if self.train_num_samples_per_epoch else None
            samplers = get_list_of_weighted_random_samplers(num_samplers=self.num_train_samplers,
                                                            dataset_weights=self.train_data_info["dataset_weights"],
                                                            dataset_index_ranges=self.train_data_info["index_ranges"],
                                                            num_samples_per_epoch=actual_train_num_samples_per_epoch)
            # Train dataloader arguments
            self.train_data_args = []
            for sampler in samplers:
                self.train_data_args.append({
                    "dataset_name": None,
                    "split": None,
                    "file_list": self.train_data_info["merged_file_list"],
                    "sub_batch_size": shared_cfg["BSZ"]["train_sub"],
                    "task_manager": self.task_manager,
                    "random_amp_range": self.train_random_amp_range,  # "0.1,0.5
                    "stem_iaug_prob": self.train_stem_iaug_prob,
                    "stem_xaug_policy": self.train_stem_xaug_policy,
                    "pitch_shift_range": self.train_pitch_shift_range,
                    "shuffle": True,
                    "sampler": sampler,
                    "audio_cfg": self.audio_cfg,
                })

            # Validation dataloader arguments
            self.val_data_args = []
            for preset_single in self.preset_singles:
                if preset_single["validation_split"] != None:
                    self.val_data_args.append({
                        "dataset_name": preset_single["dataset_name"],
                        "split": preset_single["validation_split"],
                        "task_manager": self.task_manager,
                        # "tokenizer": self.task_manager.get_tokenizer(),
                        "max_num_files": self.val_max_num_files,
                        "audio_cfg": self.audio_cfg,
                    })

        if stage == "test":
            self.test_data_args = []
            for preset_single in self.preset_singles:
                if preset_single["test_split"] != None:
                    self.test_data_args.append({
                        "dataset_name": preset_single["dataset_name"],
                        "split": preset_single["test_split"],
                        "task_manager": self.task_manager,
                        "max_num_files": self.test_max_num_files,
                        "audio_cfg": self.audio_cfg,
                    })

    def train_dataloader(self) -> Any:
        loaders = {}
        for i, args_dict in enumerate(self.train_data_args):
            loaders[f"data_loader_{i}"] = get_cache_data_loader(**args_dict, dataloader_config=shared_cfg["DATAIO"])
        return CombinedLoader(loaders, mode="min_size")  # size is always identical

    def val_dataloader(self) -> Any:
        loaders = {}
        for args_dict in self.val_data_args:
            dataset_name = args_dict["dataset_name"]
            loaders[dataset_name] = get_eval_dataloader(**args_dict, dataloader_config=shared_cfg["DATAIO"])
        return loaders

    def test_dataloader(self) -> Any:
        loaders = {}
        for args_dict in self.test_data_args:
            dataset_name = args_dict["dataset_name"]
            loaders[dataset_name] = get_eval_dataloader(**args_dict, dataloader_config=shared_cfg["DATAIO"])
        return loaders

    """CombinedLoader in "sequential" mode returns dataloader_idx to the
       trainer, which is used to get the dataset name in the logger. """

    @property
    def num_val_dataloaders(self) -> int:
        return len(self.val_data_args)

    @property
    def num_test_dataloaders(self) -> int:
        return len(self.test_data_args)

    def get_val_dataset_name(self, dataloader_idx: int) -> str:
        return self.val_data_args[dataloader_idx]["dataset_name"]

    def get_test_dataset_name(self, dataloader_idx: int) -> str:
        return self.test_data_args[dataloader_idx]["dataset_name"]
