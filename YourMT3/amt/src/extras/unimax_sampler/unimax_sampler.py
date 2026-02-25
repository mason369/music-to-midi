import torch
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, Sampler
from torch.utils.data import RandomSampler
from operator import itemgetter
from typing import List, Union, Iterator, Optional


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
    Allows you to use any sampler in distributed mode.
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


class UnimaxSampler(Sampler):
    # Initialize the sampler with the character counts for each language,
    # the total character budget, and the number of epochs per language.
    def __init__(self, language_character_counts: List[int], total_character_budget: int,
                 num_epochs: int) -> None:
        self.language_character_counts = torch.tensor(language_character_counts)
        self.total_character_budget = total_character_budget
        self.num_epochs = num_epochs
        # Compute the sampling distribution p.
        self.p = self._unimax()

    # Define how to iterate over the data. We'll use PyTorch's multinomial
    # function to generate indices according to the distribution p.
    def __iter__(self) -> iter:
        return iter(torch.multinomial(self.p, len(self.p), replacement=True).tolist())

    # Define the length of the sampler as the number of languages.
    def __len__(self) -> int:
        return len(self.p)

    # Implement the UNIMAX algorithm to compute the sampling distribution p.
    def _unimax(self) -> torch.Tensor:
        # Sort languages by character count.
        L, indices = torch.sort(self.language_character_counts)
        # Initialize the remaining budget to the total character budget.
        B = float(self.total_character_budget)
        i = 0
        # Initialize the budget per language.
        U = torch.zeros_like(L)
        # For each language...
        for idx in indices:
            # Compute the remaining budget per-language.
            bl = B / (len(L) - i)
            cl = L[idx]
            # If per-language budget exceeds N epochs of the language, use N epochs.
            if bl > cl * self.num_epochs:
                Ul = cl * self.num_epochs
            # Otherwise use uniform per-language budget.
            else:
                Ul = bl
            # Store the computed budget.
            U[idx] = Ul
            # Update the remaining budget.
            B -= Ul
            # Move to the next language.
            i += 1
        # Normalize the budget to create a distribution.
        p = U / U.sum()
        # Return the computed distribution.
        return p


class DistributedUnimaxSampler(UnimaxSampler):

    def __init__(self,
                 language_character_counts: List[int],
                 total_character_budget: int,
                 num_epochs: int,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True) -> None:

        super().__init__(language_character_counts, total_character_budget, num_epochs)
        self.distributed_sampler = DistributedSamplerWrapper(self, num_replicas, rank, shuffle)

    def __iter__(self):
        return iter(self.distributed_sampler)

    def __len__(self):
        return len(self.distributed_sampler)

    def set_epoch(self, epoch):
        self.distributed_sampler.set_epoch(epoch)