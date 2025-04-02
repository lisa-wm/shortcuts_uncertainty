"""NN architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelFactory:
    """Factory for creating NN models."""

    @staticmethod
    def create_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
        """Create a model by name."""
        match model_name:
            case 'smallconvnet':
                return SmallConvNet(num_classes=num_classes, **kwargs)
            case _:
                raise NotImplementedError(f'Model `{model_name}` not available.')


class AbstractModel(nn.Module, ABC):
    """Basic abstract model."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

    @abstractmethod
    def get_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward pass."""
        pass


class SmallConvNet(AbstractModel):
    """Really small convnet for MNIST data."""

    def __init__(
        self, num_channels: int, num_classes: int, dropout_rate: float
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=16,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 14 * 14, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward pass."""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def get_name(self) -> str:
        """Return the model name."""
        return 'smallconvnet'
