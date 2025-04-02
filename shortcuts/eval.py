"""Evaluate probabilistic predictions."""

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torchmetrics import Accuracy, CalibrationError

from shortcuts.datamodules import DataModule
from shortcuts.ensemble import EnsembleLearner
from shortcuts.utils import (
    cast_tensor_float,
    compute_entropy,
    save_as_json,
)


class Evaluator:
    """Evaluating model predictions on suitable metrics."""

    def __init__(
        self,
        ensemble: EnsembleLearner,
        dataset: DataModule,
        num_classes: int,
        top_u_percentile: Optional[float] = None,
    ) -> None:
        """Instantiate evaluator."""
        self.ensemble = ensemble
        self.dataset = dataset
        self.num_classes = num_classes
        self.entropy_upper_bound = math.log2(self.num_classes)
        self.ensemble_size = ensemble.ensemble_size
        self.metric_dict: dict = {}
        self.uncertainty_dict: dict = {}
        self.uncertainty_binned = None
        self.eu_dict: dict = {}
        self.targets = self._prepare_targets()
        self.preds = None
        self.num_obs_dict: dict = {}
        self.top_u_percentile = top_u_percentile

    def prepare_predictions(
        self, ckpt_dir: Optional[str] = None, epoch: Optional[int] = None
    ) -> None:
        """Prepare predictions."""
        preds, _ = self._prepare_predictions(ckpt_dir=ckpt_dir, epoch=epoch)
        self.preds = self._check_shape(preds)

    def evaluate_all(
        self,
        include_performance: bool = True,
        include_calibration: bool = True,
        include_uncertainty: bool = True,
        epoch: Optional[int] = None,
    ) -> None:
        """Evaluate all metrics."""
        epoch_str = 'final' if epoch is None else f'{epoch}'
        if self.preds is None:
            raise ValueError('Call `prepare_predictions` first.')
        preds = self.preds
        targets = self.targets
        if preds.shape[1] == 0:
            return
        if self.top_u_percentile is not None:
            tu = compute_entropy(preds.nanmean(0)) / self.entropy_upper_bound
            top_tu = torch.topk(tu, int(self.top_u_percentile * len(tu)), largest=True)
            preds = preds[:, top_tu.indices, :]
            targets = targets[top_tu.indices]
        dict_update = {'num_obs': len(targets)}
        if self.num_obs_dict.get(epoch_str) is None:
            self.num_obs_dict.update({epoch_str: dict_update})
        else:
            self.num_obs_dict[epoch_str].update(dict_update)

        if include_performance:
            self._evaluate_performance(preds, targets, epoch_str)
        if include_calibration:
            self._evaluate_calibration(preds, targets, epoch_str)
        if include_uncertainty:
            self._evaluate_uncertainty(preds, epoch_str)

    def _evaluate_performance(
        self, preds: torch.Tensor, targets: torch.Tensor, epoch_str: str
    ) -> None:
        """Evaluate predictive performance."""
        acc_params: dict = {
            'num_classes': self.num_classes,
            'average': 'weighted',
        }
        if self.num_classes == 2:
            acc_params.update(task='binary')
        else:
            acc_params.update(task='multiclass')
        accuracy = Accuracy(**acc_params)
        acc = accuracy(torch.argmax(preds.nanmean(0), dim=1), targets)
        dict_update = {'accuracy': cast_tensor_float(acc)}
        if self.metric_dict.get(epoch_str) is None:
            self.metric_dict.update({epoch_str: dict_update})
        else:
            self.metric_dict[epoch_str].update(dict_update)

    def _evaluate_calibration(
        self, preds: torch.Tensor, targets: torch.Tensor, epoch_str: str
    ) -> None:
        """Evaluate calibration."""
        preds = preds.nanmean(0)
        ce_params: dict = {
            'num_classes': self.num_classes,
            'n_bins': max(len(targets) // 10, 1),
        }
        if self.num_classes == 2:
            ce_params.update(task='binary')
            preds = preds[:, 1]
        else:
            ce_params.update(task='multiclass')
        ece = CalibrationError(norm='l1', **ce_params)
        maxce = CalibrationError(norm='max', **ce_params)
        dict_update = {
            'ece': cast_tensor_float(ece(preds, targets)),
            'maxce': cast_tensor_float(maxce(preds, targets)),
        }
        if self.metric_dict.get(epoch_str) is None:
            self.metric_dict.update({epoch_str: dict_update})
        else:
            self.metric_dict[epoch_str].update(dict_update)

    def _evaluate_uncertainty(self, preds: torch.Tensor, epoch_str: str) -> None:
        """Get total, epistemic, aleatoric uncertainty."""
        tu = compute_entropy(preds.nanmean(0)) / self.entropy_upper_bound
        au = compute_entropy(preds).nanmean(0) / self.entropy_upper_bound
        # handle rounding errors
        tu = tu.where(tu >= 0, torch.tensor(0.0))
        tu = tu.where(tu <= 1, torch.tensor(1.0))
        au = au.where(au >= 0, torch.tensor(0.0))
        au = au.where(au <= 1, torch.tensor(1.0))
        eu = tu - au
        eu = eu.where(eu >= 0, torch.tensor(0.0))
        eu = eu.where(eu <= 1, torch.tensor(1.0))
        dict_update = {
            epoch_str: {
                'tu': [cast_tensor_float(x) for x in tu.tolist()],
                'au': [cast_tensor_float(x) for x in au.tolist()],
                'eu': [cast_tensor_float(x) for x in eu.tolist()],
            }
        }
        if self.uncertainty_dict.get(epoch_str) is None:
            self.uncertainty_dict.update({epoch_str: dict_update})
        else:
            self.uncertainty_dict[epoch_str].update(dict_update)
        n_bins = 20
        bins = np.linspace(0, 1, n_bins + 1)
        bins = np.append(np.array([float('-inf')]), bins)
        bins = np.append(bins, np.array([float('inf')]))
        tables: list = []
        for u, uname in zip([tu, au, eu], ['tu', 'au', 'eu']):
            bin_idx = pd.cut(
                np.array(u.tolist()),
                bins=bins,
                right=False,
                labels=range(1, n_bins + 3),
            )
            frequency_table = bin_idx.value_counts()
            table = pd.DataFrame(
                {
                    'epoch': epoch_str,
                    'bin_start': bins[:-1],
                    'bin_end': bins[1:],
                    'component': uname,
                    'freq': frequency_table.values,
                }
            )
            tables.append(table)
        tables_all = pd.concat(tables, ignore_index=True)
        if self.uncertainty_binned is None:
            self.uncertainty_binned = tables_all
        else:
            self.uncertainty_binned = pd.concat(
                [self.uncertainty_binned, tables_all], ignore_index=True
            )

    def get_num_obs_dict(self) -> dict:
        """Get number of observations."""
        return self.num_obs_dict

    def get_metric_dict(self) -> dict:
        """Get metric dictionary."""
        return self.metric_dict

    def get_uncertainty_dict(self) -> dict:
        """Get uncertainty dictionary."""
        return self.uncertainty_dict

    def get_uncertainty_binned(self) -> pd.DataFrame:
        """Get df with binned uncertainties."""
        return self.uncertainty_binned

    def save_num_obs_dict(self, save_to: str) -> None:
        """Save number of observations."""
        save_as_json(self.get_num_obs_dict(), save_to)

    def save_metric_dict(self, save_to: str) -> None:
        """Save metric dictionary."""
        save_as_json(self.get_metric_dict(), save_to)

    def save_uncertainty_dict(self, save_to: str) -> None:
        """Save uncertainty dictionary."""
        save_as_json(self.get_uncertainty_dict(), save_to)

    def save_uncertainty_binned(self, save_to: str) -> None:
        """Save df with binned uncertainties."""
        u_binned = self.get_uncertainty_binned()
        if u_binned is not None:
            self.get_uncertainty_binned().to_csv(save_to, index=False)

    def _check_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Check shape of input."""
        if not x.shape == (self.ensemble_size, x.shape[1], self.num_classes):
            raise ValueError('Predictions have wrong shape.')
        return x

    def _prepare_targets(self) -> torch.Tensor:
        """Get test targets."""
        self.dataset.setup(stage='test')  # get all targets in single batch
        _, targets = next(iter(self.dataset.test_dataloader()))
        return targets

    def _prepare_predictions(
        self,
        ckpt_dir: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Prepare predictions and targets."""
        predictions = self.ensemble.predict(
            self.dataset,
            avg=False,
            ckpt_dir=ckpt_dir,
            epoch=epoch,
        )
        predictions = self._check_shape(predictions)
        return predictions
