"""Data modules for experiments."""

import random
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Optional,
    Union,
)

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from shortcuts.config import TransformConfig
from shortcuts.data import (
    ColoredMNIST,
    CustomMNIST,
    ShortcutDataset,
    ThreewayMNIST,
    ZeroMNIST,
)


class DataModule(ABC, LightningDataModule):
    """Base class for data modules."""

    def __init__(
        self,
        dataset_name: str,
        save_to: str,
        transform_base: list,
        transform_train: list,
        transform_test: list,
        batch_size_train: int,
        batch_size_test: int,
        val_size: float = 0.2,
        num_obs: Optional[int] = None,
        processing_fun: Optional[Callable] = None,
        processing_args: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> None:
        """Instantiate data module."""
        super().__init__()
        self.dataset_name = dataset_name
        self.save_to = save_to
        self.val_size = val_size
        self.num_obs = num_obs
        self.processing_fun = processing_fun
        self.processing_args = processing_args
        self.transform_base = transform_base
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.data_train: Union[datasets.VisionDataset, None] = None
        self.data_val: Union[datasets.VisionDataset, None] = None
        self.data_pred: Union[datasets.VisionDataset, None] = None
        self.data_test: Union[datasets.VisionDataset, None] = None
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare data."""
        pass

    @abstractmethod
    def setup(self, stage: str) -> None:
        """Assign data for dataloaders."""
        pass

    def train_dataloader(self) -> DataLoader:
        """Set up train dataloader."""
        return DataLoader(self.data_train, batch_size=self.batch_size_train)

    def val_dataloader(self) -> DataLoader:
        """Set up validation dataloader."""
        return DataLoader(self.data_val, batch_size=self.batch_size_train)

    def predict_dataloader(self) -> DataLoader:
        """Set up test dataloader."""
        return DataLoader(
            self.data_pred,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        """Set up test dataloader."""
        batch_size = len(self.data_test) if self.data_test is not None else 32
        return DataLoader(self.data_test, batch_size=batch_size, shuffle=False)


class MNISTLikeModule(DataModule, ABC):
    """Data module for MNIST-like datasets."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Init MNIST data module."""
        super().__init__(*args, **kwargs)
        self.data_args: dict = {**kwargs}

    def prepare_data(self) -> None:
        """Prepare data."""
        pass

    def setup(self, stage: str, **kwargs) -> None:
        """Assign data for dataloaders."""
        match stage:
            case 'fit':
                data_full = self._get_data(
                    is_train=True,
                    transform=transforms.Compose(
                        self.transform_train + self.transform_base
                    ),
                )
                if self.num_obs is not None:
                    data_full = self._subsample(data_full, self.num_obs)
                processing_fun = self.processing_fun or (lambda x: x)
                processing_args = self.processing_args
                data_full = self._postprocess_data(
                    data_full,
                    processing_fun,
                    processing_args,
                )
                num_val = int(len(data_full) * self.val_size)
                self.data_train, self.data_val = random_split(
                    data_full,
                    lengths=[len(data_full) - num_val, num_val],
                    generator=torch.Generator().manual_seed(42),
                )
            case 'predict':
                data = self._get_data(
                    is_train=False,
                    transform=transforms.Compose(
                        self.transform_test + self.transform_base
                    ),
                )
                processing_fun = self.processing_fun or (lambda x: x)
                processing_args = self.processing_args
                self.data_pred = self._postprocess_data(
                    data,
                    processing_fun,
                    processing_args,
                )
            case 'test':
                self.data_test = self._get_data(
                    is_train=False,
                    transform=transforms.Compose(
                        self.transform_test + self.transform_base
                    ),
                )
            case _:
                raise ValueError(f'Unknown stage `{stage}` of lightning module')

    @staticmethod
    def _subsample(data_full: ShortcutDataset, num_obs: int) -> ShortcutDataset:
        """Subsample data."""
        if num_obs is not None:
            if num_obs > len(data_full):
                raise ValueError('Number of observations exceeds dataset size.')
            idx = random.sample(range(len(data_full)), num_obs)
            data_full.data = data_full.data[idx]
            data_full.targets = data_full.targets[idx]
        return data_full

    @abstractmethod
    def _get_data(
        self, is_train: bool, transform: transforms.Compose, *args, **kwargs
    ) -> ShortcutDataset:
        """Get raw data."""
        pass

    @staticmethod
    def _postprocess_data(
        dataset: ShortcutDataset,
        processing_fun: Callable,
        processing_args: Optional[dict] = None,
    ) -> ShortcutDataset:
        """Postprocess data."""
        processing_args = processing_args or {}
        return processing_fun(dataset, **processing_args)


class MNISTModule(MNISTLikeModule):
    """Data module for MNIST."""

    def _get_data(
        self, is_train: bool, transform: transforms.Compose, *args, **kwargs
    ) -> ShortcutDataset:
        """Get raw data."""
        return CustomMNIST(
            self.save_to,
            train=is_train,
            transform=transform,
        )


class ThreewayMNISTModule(MNISTLikeModule):
    """Data module for MNIST with 3 classes (pooling classes of original MNIST)."""

    def __init__(
        self,
        num_channels: int,
        patch: bool = False,
        patch_prob: float = 0.0,
        *args,
        **kwargs,
    ):
        """Init Threeway MNIST data module."""
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.patch = patch
        self.patch_prob = patch_prob

    def _get_data(
        self, is_train: bool, transform: transforms.Compose, *args, **kwargs
    ) -> ShortcutDataset:
        """Get raw data."""
        return ThreewayMNIST(
            self.save_to,
            train=is_train,
            transform=transform,
            num_channels=self.num_channels,
            patch=self.patch,
            patch_prob=self.patch_prob,
        )


class ZeroMNISTModule(MNISTLikeModule):
    """Data module for MNIST with only class 0."""

    def __init__(self, num_channels: int, *args, **kwargs):
        """Init Zero MNIST data module."""
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    def _get_data(
        self, is_train: bool, transform: transforms.Compose, *args, **kwargs
    ) -> ShortcutDataset:
        """Get raw data."""
        return ZeroMNIST(
            self.save_to,
            train=is_train,
            transform=transform,
            num_channels=self.num_channels,
        )


class ColoredMNISTModule(MNISTLikeModule):
    """Data module for colored MNIST."""

    def __init__(
        self,
        shortcut_strength: int,
        num_classes: int,
        *args,
        **kwargs,
    ) -> None:
        """Init Colored MNIST data module."""
        super().__init__(*args, **kwargs)
        self.data_args: dict = {
            'shortcut_strength': shortcut_strength,
            'num_classes': num_classes,
        }

    def _get_data(
        self, is_train: bool, transform: transforms.Compose, *args, **kwargs
    ) -> ShortcutDataset:
        """Get raw data."""
        return ColoredMNIST(
            self.save_to,
            train=is_train,
            transform=transform,
            shortcut_strength=self.data_args.get('shortcut_strength', 0.0),
            num_classes=self.data_args.get('num_classes', 2),
        )


class DataModuleFactory:
    """Factory for creating data modules."""

    @staticmethod
    def create_module(
        dataset_name: str,
        save_to: str,
        batch_size_train: int,
        batch_size_test: int,
        val_size: float = 0.2,
        num_obs: Optional[int] = None,
        **kwargs,
    ) -> DataModule:
        """Create a lightning data module by name."""
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        rotate = transforms.RandomRotation(degrees=10)
        affine = transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        )
        transform_config = TransformConfig(
            base=[to_tensor, normalize], train=[rotate, affine], test=[]
        )
        mandatory_args: dict = {
            'dataset_name': dataset_name,
            'save_to': save_to,
            'val_size': val_size,
            'batch_size_train': batch_size_train,
            'batch_size_test': batch_size_test,
            'transform_base': transform_config.base,
            'transform_train': transform_config.train,
            'transform_test': transform_config.test,
            'num_obs': num_obs,
        }
        match dataset_name:
            case 'mnist':
                return MNISTModule(**mandatory_args, **kwargs)
            case 'mnist0':
                return ZeroMNISTModule(**mandatory_args, **kwargs)
            case 'mnist3' | 'pmnist3' | 'mnist3c':
                return ThreewayMNISTModule(**mandatory_args, **kwargs)
            case 'cmnist2' | 'cmnist3':
                return ColoredMNISTModule(**mandatory_args, **kwargs)
            case _:
                raise NotImplementedError(f'Dataset `{dataset_name}` not available.')
