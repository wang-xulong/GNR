import torch
import torch.nn as nn
from avalanche.models.dynamic_modules import MultiTaskModule, MultiHeadClassifier, IncrementalClassifier


class Alexnet(nn.Module):
    """
    Simplified AlexNet model for single-task classification.
    """

    def __init__(self, input_size=(3, 32), num_classes=100, drop1=0.5, drop2=0.5):
        super(Alexnet, self).__init__()
        nch, size = input_size[0], input_size[1]

        # Feature extraction layers (Convolution + Fully Connected)
        self.conv_features = nn.Sequential(
            nn.Conv2d(nch, 64, kernel_size=size // 8),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=size // 10),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=2),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop2),
            nn.MaxPool2d(2),
        )

        # Compute flattened feature size
        smid = self.compute_conv_output_size(size, size // 8) // 2
        smid = self.compute_conv_output_size(smid, size // 10) // 2
        smid = self.compute_conv_output_size(smid, 2) // 2
        self.flattened_size = 256 * smid * smid

        self.fc_features = nn.Sequential(
            nn.Linear(self.flattened_size, 2048),
            # nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop2),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop2),
        )

        # Classification layer
        self.classifier = nn.Linear(2048, num_classes)

    @staticmethod
    def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
        return int((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)

        # Fully connected feature extraction
        x = self.fc_features(x)

        # Classification
        x = self.classifier(x)
        return x

    # 新增的方法，用于提取特征提取器和分类器
    def get_feature_extractor(self):
        return nn.Sequential(self.conv_features,
                             nn.Flatten(),
                             self.fc_features)

    def get_classifier(self):
        return self.classifier


class ICAlexnet(Alexnet, MultiTaskModule):
    """
    Multi-task version of AlexNet using MultiHeadClassifier.
    """

    def __init__(self, input_size=(3, 32), num_classes=100, drop1=0.5, drop2=0.5):
        super().__init__(input_size, num_classes, drop1, drop2)
        # Replace single-task classifier with multi-head classifier
        self.classifier = IncrementalClassifier(in_features=2048,
                                                # initial_out_features=initial_out_features,
                                                # masking=False,
                                                )

    def forward(self, x, task_labels):
        # Use inherited feature extraction from Alexnet
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x)

        # Multi-task classification
        x = self.classifier(x)
        return x


class MTAlexnet(Alexnet, MultiTaskModule):
    """
    Multi-task version of AlexNet using MultiHeadClassifier.
    """

    def __init__(self, input_size=(3, 32), num_classes=100, drop1=0.5, drop2=0.5):
        super().__init__(input_size, num_classes, drop1, drop2)
        # Replace single-task classifier with multi-head classifier
        self.classifier = MultiHeadClassifier(in_features=2048,
                                              # initial_out_features=initial_out_features,
                                              masking=False,
                                              )

    def forward(self, x, task_labels):
        # Use inherited feature extraction from Alexnet
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x)

        # Multi-task classification
        x = self.classifier(x, task_labels)
        return x


__all__ = ["Alexnet", "MTAlexnet", "ICAlexnet"]
