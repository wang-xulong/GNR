import torch.nn as nn
import numpy as np

from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)


class Alexnet(nn.Module):
    def __init__(
            self,
            num_classes=10,
            hidden_size=100,
            drop1=0.1,
            drop2=0.1,
            input_size=None):
        super(Alexnet, self).__init__()
        if input_size is None:
            raise ValueError('input_size must be specified as an list [channels, height, width]')

        nch, size = input_size[0], input_size[1]

        self.c1 = nn.Conv2d(nch, 64, kernel_size=size // 8)
        s = self.compute_conv_output_size(size, size // 8)
        s = s // 2

        self.c2 = nn.Conv2d(64, 128, kernel_size=size // 10)
        s = self.compute_conv_output_size(s, size // 10)
        s = s // 2

        self.c3 = nn.Conv2d(128, 256, kernel_size=2)
        s = self.compute_conv_output_size(s, 2)
        s = s // 2

        self.smid = s
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)
        self.fc1 = nn.Linear(256 * self.smid ** 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.contiguous()
        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))

        h = h.view(h.shape[0], -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))

        h = self.classifier(h)
        return h

    @staticmethod
    def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1) -> int:
        return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


# enddef
class MTAlexnet(Alexnet, MultiTaskModule):

    def __init__(self,
                 num_classes=10,
                 hidden_size=100,
                 drop1=0.1,
                 drop2=0.1,
                 input_size=None):
        super().__init__(num_classes=num_classes,
                         hidden_size=hidden_size,
                         drop1=drop1,
                         drop2=drop2,
                         input_size=input_size)

        self.classifier = MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        x = x.contiguous()
        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))

        h = h.view(h.shape[0], -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        h = h.squeeze()
        h = self.classifier(h, task_labels)
        return h


__all__ = ["Alexnet", "MTAlexnet"]

if __name__ == "__main__":
    from avalanche.benchmarks import SplitMNIST, SplitCIFAR100

    benchmark = SplitMNIST(5, shuffle=False, return_task_id=True, class_ids_from_zero_in_each_exp=True)

    inputsize = (3, 32)
    a = MTAlexnet(hidden_size=100, drop1=0.5, drop2=0.5, input_size=inputsize)

    for exp in benchmark.train_stream:
        a.adaptation(exp)
        print(a)
