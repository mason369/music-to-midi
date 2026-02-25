# UniMax Language Dataset Sampler with DDP support

This repository contains an unofficial implementation of the UNIMAX sampling algorithm using PyTorch. The UNIMAX algorithm ["UniMax: Fairer and more Effective Language Sampling for Large-Scale Multilingual Pretraining" by HW Chung et al. (ICLR 2023)](https://arxiv.org/abs/2304.09151) is used to generate a sampling distribution of languages based on their character counts, a total character budget, and a specified number of epochs per language. This can be useful for training language models on datasets with imbalanced language distribution.

## Contents

1. `unimax_sampler.py`: This Python file contains the `UnimaxSampler` class, a PyTorch `Sampler` that uses the UNIMAX algorithm.

2. `test_unimax_sampler.py`: This Python file contains a unit test for the `UnimaxSampler` class to ensure its correct functionality.

## Usage

```python
from torch.utils.data import Dataset, DataLoader
from unimax_sampler import UnimaxSampler

# Define your parameters
language_character_counts = [100, 200, 300, 400, 500]
total_character_budget = 1000
num_epochs = 2

# Create the UnimaxSampler
unimax_sampler = UnimaxSampler(language_character_counts, total_character_budget, num_epochs)
```

Then, use the sampler as the sampler argument when creating a DataLoader.

```python
# Disable shuffle when using custom sampler...
data_loader = DataLoader(my_dataset, batch_size=2, shuffle=None, sampler=unimax_sampler)
```

For DDP,
```python
if torch.distributed.is_initialized():
    sampler = DistributedUnimaxSampler(...)
else:
    return unimax_sampler(...)
```

## Note
The initial version of this code was created by [Chat GPT-4](https://chat.openai.com/), based on the pseudocode provided in the [UNIMAX](https://arxiv.org/abs/2304.09151) paper. Subsequently, the code was manually revised for `PyTorch` Distributed Data Parallel ([DDP](https://pytorch.org/docs/stable/notes/ddp.html)) framework. The DistributedSamplerWrapper implementation is derived from an earlier version found in the [Catalyst](https://github.com/catalyst-team/catalyst) project.

## License
This project is licensed under the MIT License.