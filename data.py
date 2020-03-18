import torch
import numpy as np
from torch.utils.data import Dataset


class SimpleSine(Dataset):
    def __init__(self, num_samples=1000, num_points=100):
        self.num_samples = num_samples
        self.num_points = num_points
        self.data = []

        amp_max, amp_min = 1, -1
        shift_max, shift_min = 0.5, -0.5

        x = torch.linspace(-np.pi, np.pi, self.num_points).unsqueeze(1)

        # generate data
        for _ in range(num_samples):

            amplitude = (amp_max - amp_min) * np.random.randn() + amp_min
            shift = (shift_max - shift_min) * np.random.randn() + shift_min
            y = amplitude * torch.sin(x - shift)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
