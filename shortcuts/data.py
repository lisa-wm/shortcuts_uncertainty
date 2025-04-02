"""Datasets for experiments."""

import os
from typing import (
    Callable,
    Final,
    Optional,
)

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision.datasets import MNIST, VisionDataset

from root_path import ROOT

DATA_DIR: Final[str] = os.path.join(ROOT, 'datasets/')
LABEL_DUMMY: Final[int] = -1


class ShortcutDataset(VisionDataset):
    """Abstract class for datasets."""

    data: torch.tensor
    targets: torch.tensor

    def __init_subclass__(cls, **kwargs):
        """Make sure that subclasses define data and targets."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'data'):
            raise TypeError(
                f"Class {cls.__name__} must define a 'data' class attribute."
            )
        if not hasattr(cls, 'targets'):
            raise TypeError(
                f"Class {cls.__name__} must define a 'targets' class attribute."
            )


class CustomMNIST(MNIST, ShortcutDataset):
    """Custom MNIST."""

    data = torch.empty(0)
    targets = torch.empty(0)

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Init BinaryMNIST."""
        MNIST.__init__(
            self,
            root=root,
            download=True,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )


class ThreewayMNIST(CustomMNIST):
    """MNIST with fewer classes."""

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_channels: int = 1,
        patch: bool = False,
        patch_prob: float = 0.5,
    ) -> None:
        """Init BinaryMNIST."""
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )
        self.save_to = root
        self.num_channels = num_channels
        self.patch = patch
        self.patch_prob = patch_prob
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data."""
        self.targets = torch.tensor(
            [map_labels(i, num_classes=3) for i in self.targets]
        )
        idx = self.targets != LABEL_DUMMY
        match self.num_channels:
            case 1:
                pass
            case 3:
                grayscale_images = self.data.unsqueeze(3)
                self.data = grayscale_images.expand(-1, -1, -1, 3)
            case _:
                raise NotImplementedError('Only 1 or 3 channels supported.')
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        img, label = self.data[index], self.targets[index]
        img_mode = 'L' if self.num_channels == 1 else 'RGB'
        img = Image.fromarray(img.numpy(), mode=img_mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.patch and np.random.uniform() < self.patch_prob:
            if self.num_channels == 1:
                patch_color = float(torch.max(img))
                match label:
                    case 0:
                        img[:, 1:3, 1:3] = patch_color
                    case 1:
                        img[:, 1:3, -3:-1] = patch_color
                    case 2:
                        img[:, -3:-1, 1:3] = patch_color
                    case _:
                        pass
            elif self.num_channels == 3:
                arr = np.array(img)
                match label:
                    case 0:
                        pos = 0
                    case 1:
                        pos = 1
                    case 2:
                        pos = 2
                    case _:
                        raise ValueError('Invalid label.')
                arr[pos, 1, 1] = 1
                img = torch.tensor(arr)
            else:
                raise NotImplementedError('Only 1 or 3 channels supported.')
        return img, label


class ZeroMNIST(CustomMNIST):
    """MNIST with 0 class only."""

    def __init__(
        self,
        root: str,
        train: bool,
        num_channels: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Init BinaryMNIST."""
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )
        self.save_to = root
        self.num_channels = num_channels
        self._prepare_data()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        img, label = self.data[index], self.targets[index]
        match self.num_channels:
            case 1:
                img = Image.fromarray(img.numpy(), mode='L')
            case 3:
                img = Image.fromarray(img.numpy(), mode='RGB')
            case _:
                raise NotImplementedError('Only 1 or 3 channels supported.')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def _prepare_data(self) -> None:
        """Prepare data."""
        idx = self.targets == 0
        self.data = self.data[idx]
        match self.num_channels:
            case 1:
                pass
            case 3:
                grayscale_images = self.data.unsqueeze(3)
                self.data = grayscale_images.expand(-1, -1, -1, 3)
            case _:
                raise NotImplementedError('Only 1 or 3 channels supported.')
        self.targets = torch.zeros_like(self.targets[idx]) + LABEL_DUMMY


class ColoredMNIST(ShortcutDataset):
    """Colored MNIST."""

    data = torch.empty(0)
    targets = torch.empty(0)

    def __init__(
        self,
        root: str,
        shortcut_strength: float,
        num_classes: int,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Init ColoredMNIST."""
        assert num_classes in [2, 3]
        assert 0 <= shortcut_strength <= 1

        VisionDataset.__init__(
            self, root, transform=transform, target_transform=target_transform
        )
        self.save_to = root
        self.shortcut_strength = shortcut_strength
        self.num_classes = num_classes
        self.env = 'train' if train else 'test'
        self._prepare_data()
        data_file = f'{os.path.join(self.root, self.env)}.pt'
        data_tuples_all = torch.load(data_file, weights_only=True)
        os.unlink(data_file)
        self.data = torch.stack([i[0] for i in data_tuples_all])
        self.targets = torch.stack([i[1] for i in data_tuples_all])
        idx = self.targets != LABEL_DUMMY
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self) -> int:
        """Get number of observations."""
        return len(self.targets)

    def _prepare_data(self) -> None:
        """Prepare coloring."""
        mnist_data = datasets.MNIST(
            download=True,
            root=self.save_to,
            train=self.env == 'train',
        )

        def map_colors(label_int: int) -> str:
            match label_int:
                case 0:
                    return 'red'
                case 1:
                    return 'green'
                case 2:
                    return 'blue'
                case _:
                    return ''

        def color_grayscale_arr(arr: np.array, color: str) -> np.array:
            """Convert grayscale img to red or green."""
            assert arr.ndim == 2
            dtype = arr.dtype
            h, w = arr.shape
            arr = np.reshape(arr, [h, w, 1])
            match color:
                case 'red':
                    arrays = [arr, np.zeros((h, w, 2), dtype=dtype)]  # g, b zero
                case 'green':  # r, b zero
                    arrays = [
                        np.zeros((h, w, 1), dtype=dtype),
                        arr,
                        np.zeros((h, w, 1), dtype=dtype),
                    ]
                case 'blue':  # r, g zero
                    arrays = [
                        np.zeros((h, w, 2), dtype=dtype),
                        arr,
                    ]
                case _:
                    arrays = [arr, arr, arr]
            return np.concatenate(arrays, axis=2)

        inst_list: list = []
        cutoff_perturb = 1 - self.shortcut_strength

        if self.env == 'train':
            for idx in range(len(mnist_data)):
                np.random.seed(123 + idx)
                im, label = mnist_data[idx]
                im_array = np.array(im)
                color_label = map_labels(label, self.num_classes)
                color_str = map_colors(color_label)
                if np.random.uniform() < cutoff_perturb:
                    color_str = ''
                colored_arr = color_grayscale_arr(im_array, color_str)
                inst_list.append((torch.tensor(colored_arr), torch.tensor(color_label)))

        else:
            for idx in range(len(mnist_data)):
                im, label = mnist_data[idx]
                colored_arr = color_grayscale_arr(np.array(im), '')
                color_label = map_labels(label, self.num_classes)
                inst_list.append((torch.tensor(colored_arr), torch.tensor(color_label)))

        torch.save(inst_list, os.path.join(self.save_to, f'{self.env}.pt'))


def map_labels(old_label: int, num_classes: int) -> int:
    """Map MNIST labels to two or three classes."""
    if num_classes == 2:
        match old_label:
            case 1 | 2 | 3 | 4 | 5:
                return 0
            case 6 | 7 | 8 | 9:
                return 1
            case _:
                return LABEL_DUMMY
    else:
        match old_label:
            case 1 | 2 | 3:
                return 0
            case 4 | 5 | 6:
                return 1
            case 7 | 8 | 9:
                return 2
            case _:
                return LABEL_DUMMY
