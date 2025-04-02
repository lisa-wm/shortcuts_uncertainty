"""Data classes for configuration."""

from dataclasses import dataclass

import yaml  # type: ignore[import]
from torch import nn


class ExperimentConfig:
    """Configuration class."""

    def __init__(self, path_to_config: str, experiment_id: str, seed: int) -> None:
        """Set up configuration."""
        with open(path_to_config, 'r') as file:
            base_config = yaml.safe_load(file)
        self.experiment_id = experiment_id
        self.seed = seed
        self.config_data = base_config.get('data')
        self.config_model = base_config.get('model')
        self.config_training = base_config.get('training')
        self.config: dict = {'base_config': base_config}
        self.processing: dict = {}
        self.processing_type = ''

    def to_yaml(self, timestamp: str, path: str) -> None:
        """Save configuration to YAML file."""
        log_dict = {'timestamp': timestamp, **self.config}
        if log_dict.get('processing') is not None:
            log_dict['processing'] = self.processing_type
        with open(path, 'w') as file:
            yaml.safe_dump(log_dict, file)

    def get_config(self) -> dict:
        """Get configuration."""
        return self.config

    def get_data_specs(self) -> dict:
        """Get data specs."""
        data_specs: dict = {
            'num_classes': self.config_data['num_classes'],
            'batch_size_train': self.config_data['batch_size']['train'],
            'batch_size_test': self.config_data['batch_size']['test'],
            **self.processing,
        }
        num_obs = self.config_data.get('num_obs')
        if num_obs is not None:
            data_specs.update({'num_obs': num_obs})
        return data_specs

    def get_inherent_sc_strengths(self) -> list[float]:
        """Get list of SC strengths for data with inherent SC."""
        sc_strengths = self.config_data.get('inherent_shortcut_strength')
        if sc_strengths is None:
            raise ValueError('No inherent SC strength specified in config file.')
        return sc_strengths


@dataclass
class TransformConfig:
    """Configuration for data transformations."""

    base: list
    train: list
    test: list


@dataclass
class PathConfig:
    """Configuration for dir paths."""

    config: str
    data: str
    checkpoints: str
    results: str


@dataclass
class BaseLearnerConfig:
    """Configuration for BaseLearner class."""

    device: str
    model: nn.Module
    hyperparams: dict
    num_classes: int
    log_wandb: bool
    seed: int = 1
