import torch
import torch.utils.data
import torchvision
import numpy as np

class EchoDataSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, max_class_difference=None, balance=False, n_bins=None, bins=None, num_samples=None, limits=None, **kwargs):
        if limits:
            self.min_label = limits[0]
            self.max_label = limits[1]
        else:
            self.min_label, self.max_label = self._get_min_max(dataset)
        if balance:
            if bins is not None:
                self.bins = np.array(bins)
            else:
                self.bins = np.linspace(self.min_label, self.max_label, n_bins+1)[1:]

        self.indices = list(range(len(dataset)))


        if balance:
            label_to_count = {}
            for idx in self.indices:
                label = self._get_label(dataset, idx)
                if label is None:
                    continue
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
            weights = [1.0 / label_to_count[self._get_label(dataset, idx)] 
                       if self._get_label(dataset, idx) is not None else 0
                        for idx in self.indices]
            smallest_class = min(label_to_count.values())
        else:
            weights = [1 for _ in self.indices]

        self.weights = torch.DoubleTensor(weights)
        
        if num_samples is None and balance:
            self.num_samples = sum([min(v, smallest_class*max_class_difference) for _, v in label_to_count.items()])
        elif num_samples is None:
            self.num_samples = len(dataset)
        else:
            self.num_samples = num_samples

        if not balance or self.num_samples // len(label_to_count.keys()) < smallest_class:
            self.replacement = False
        else: self.replacement = True


    def _get_label(self, dataset, idx):
        label_ = np.mean(dataset.get_label(idx))
        if label_ < self.min_label or label_ > self.max_label:
            return None
        return np.argmax(self.bins>=label_)
        
    def _get_min_max(self, dataset):
        label_list = [np.mean(dataset.get_label(idx)) for idx in range(len(dataset))]
        return np.min(label_list), np.max(label_list)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement))