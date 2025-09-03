import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, sampler=None):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        self.drop_last = True

        if sampler is not None:
            self.shuffle = False

        self.init_kwargs = {
            'dataset': self.dataset,
            'drop_last': self.drop_last,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(BaseDataLoader, self).__init__(sampler=sampler, **self.init_kwargs)

    def _split_sampler(self, split):

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)

        return train_sampler


class Dataloader(BaseDataLoader):
    def __init__(self, batch_size, num_workers, shuffle, dataset):
        self.dataset = dataset
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        super(Dataloader, self).__init__(self.dataset, batch_size, shuffle, num_workers,
                                         sampler=self.sampler)
