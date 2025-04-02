"""Ensembles of base learners."""

import math
import os
import os.path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
from torch.nn.functional import one_hot

from shortcuts.datamodules import DataModule
from shortcuts.models import AbstractModel
from shortcuts.utils import compute_entropy, init_weights


class BaseLearner(pl.LightningModule):
    """Vanilla network training with PyTorch Lightning."""

    def __init__(
        self,
        device: str,
        model: AbstractModel,
        hyperparams: dict,
        num_classes: int,
        log_wandb: bool,
        seed: int = 1,
    ) -> None:
        """Set up learner object."""
        super().__init__()
        self._device = device
        self.model = model.to(self._device)
        self.model_name = self.model.get_name()
        self.log_wandb = log_wandb
        self.seed = seed
        self.num_classes = num_classes
        self.optimizer = None
        self.scheduler = None
        self.hyperparams = hyperparams
        self.save_hyperparameters()

        acc_params: dict = {
            'num_classes': self.num_classes,
            'average': 'weighted',
        }

        if self.num_classes == 2:
            self.lossfun_train = nn.CrossEntropyLoss()
            self.lossfun_val = nn.CrossEntropyLoss()
            self.acc_train = torchmetrics.Accuracy(
                task='binary',
                **acc_params,
            )
            self.acc_val = torchmetrics.Accuracy(
                task='binary',
                **acc_params,
            )
        else:
            self.lossfun_train = nn.BCEWithLogitsLoss(reduction='sum')
            self.lossfun_val = nn.BCEWithLogitsLoss(reduction='sum')
            self.acc_train = torchmetrics.Accuracy(
                task='multiclass',
                **acc_params,
            )
            self.acc_val = torchmetrics.Accuracy(task='multiclass', **acc_params)
        self.loss_train = torchmetrics.MeanMetric()
        self.loss_val = torchmetrics.MeanMetric()
        self.log_every_u = self.hyperparams.get('log_freq_uncertainty', 10)
        self.entropies: dict = {}

    def on_fit_start(self) -> None:
        """Set global seed."""
        pl.seed_everything(seed=self.seed)

    def configure_optimizers(self) -> dict:
        """Set up optimization-related objects."""
        if self.hyperparams.get('optimizer') == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hyperparams.get('learning_rate', 0.001),
                weight_decay=self.hyperparams.get('weight_decay', 0.01),
            )
        else:
            raise ValueError('Unknown optimizer')
        if self.hyperparams.get('lr_scheduler') == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=self.hyperparams.get('lr_patience', 10),
                threshold=0.0001,
                threshold_mode='abs',
            )
        else:
            raise ValueError('Unknown scheduler')
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {'scheduler': self.scheduler, 'monitor': 'loss_val'},
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define standard forward pass."""
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Define training routine."""
        x, y = batch
        preds = self.model(x)
        targets = self._expand_targets(y)
        loss = self.lossfun_train(preds, targets.float())
        self.loss_train.update(loss.detach())
        self.acc_train.update(preds, targets)
        self.log('loss_train', self.loss_train.compute(), on_epoch=True)
        self.log('acc_train', self.acc_train.compute(), on_epoch=True)
        self.loss_train.reset()
        self.acc_train.reset()
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset pred logger."""
        epoch = self.current_epoch
        self.log('epoch', epoch)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Define validation routine."""
        # standard metric computation
        x, y = batch
        preds = self.model(x)
        targets = self._expand_targets(y)
        loss = self.lossfun_val(preds, targets.float())
        self.loss_val.update(loss.detach())
        self.acc_val.update(preds, targets)
        self.log('loss_val', self.loss_val.compute(), on_epoch=True)
        self.log('acc_val', self.acc_val.compute(), on_epoch=True)
        self.loss_val.reset()
        self.acc_val.reset()

        # custom logging
        epoch = self.current_epoch
        # example images
        if epoch == 0 and batch_idx == 0:
            if hasattr(self.logger, 'log_image'):
                self.logger.log_image(
                    key='example_image',
                    images=[x[i] for i in range(len(y))],
                    caption=[f'example img for class {i}' for i in y.tolist()],
                )
        # entropy logging
        elif epoch + 1 >= self.log_every_u and (epoch + 1) % self.log_every_u == 0:
            entropy_upper_bound = math.log2(self.num_classes)
            entropies = torch.round(
                compute_entropy(preds) / entropy_upper_bound, decimals=4
            )
            table_data = [[idx, e] for idx, e in enumerate(entropies.tolist())]
            self.entropies.update({epoch: table_data})
            self.intermediate_preds.update(
                {epoch: torch.cat((self.intermediate_preds[epoch], preds), dim=0)}
            )

        return loss

    def on_train_end(self) -> None:
        """Log entropy values."""
        entropies_tbl: list = []
        for k in self.entropies.keys():
            entropies_tbl.extend([[k] + vi for vi in self.entropies[k]])
        if hasattr(self.logger, 'log_table'):
            self.logger.log_table(
                'entropies', data=entropies_tbl, columns=['epoch', 'idx', 'entropy']
            )
        if self.log_wandb:
            for k in self.entropies.keys():
                wandb.log(
                    {
                        f'histogram_epoch_{k: 02d}': wandb.plot.histogram(
                            wandb.Table(
                                data=self.entropies[k], columns=['idx', 'entropy']
                            ),
                            value='entropy',
                            title=f'entropy in epoch {k}',
                        )
                    }
                )

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx=0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Define prediction routine."""
        x, y = batch
        return self.model(x), self.model(x)

    def init_model_weights(
        self, pretraining: bool = False, seed: Optional[int] = None
    ) -> None:
        """Initialize model weights."""
        torch.manual_seed(seed or self.seed)
        self.model.apply(lambda m: init_weights(m, pretraining=pretraining))

    def get_model(self) -> nn.Module:
        """Get model backbone."""
        return self.model

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes

    def set_seed(self, seed: int) -> None:
        """Set seed for reproducibility."""
        self.seed = seed

    def _expand_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Expand scalar class labels to one-hot encoding."""
        if len(y.shape) == 1:
            targets = one_hot(y, num_classes=self.num_classes)
        else:
            targets = y
        return targets


class EnsembleLearner:
    """Ensemble of base learners."""

    def __init__(
        self,
        device: str,
        ensemble_id: str,
        learners: list[BaseLearner],
        hyperparams: dict,
        log_wandb: bool = True,
        seed: int = 1,
        ckpt: Optional[str] = None,
    ) -> None:
        """Set up ensemble object."""
        self.ensemble_id = ensemble_id
        self.learners = learners
        self.ensemble_size = len(learners)
        self.num_classes = set([i.get_num_classes() for i in learners])
        if len(self.num_classes) != 1:
            raise ValueError('Number of classes must be the same for all learners.')
        self.hyperparams = hyperparams
        self.trainers: list = []
        self.intermediate_preds_ensemble: dict = {}
        self.log_wandb = log_wandb
        self.device = device
        self.ckpt = ckpt
        self.seed = seed
        self._init_ensemble()

    def train(
        self,
        dm: DataModule,
        epochs: int,
        device: str = 'cpu',
        num_devices: int = 1,
    ) -> None:
        """Train ensemble."""
        for idx, bl in enumerate(self.learners):
            if self.log_wandb:
                wandb.init(
                    project='shortcuts',
                    tags=[
                        f'{dm.dataset_name}',
                        f'bs{dm.batch_size_train}',
                        f'{self.learners[0].model_name}',
                    ],
                )
                logger = WandbLogger(
                    name=f'{self.ensemble_id}_{idx}',
                    log_freq=self.hyperparams.get('log_freq', 10),
                )
            else:
                logger = None
            if self.ckpt is not None:
                ckpt = os.path.join(self.ckpt, f'member_{idx}')
                os.makedirs(ckpt)
                callbacks = [
                    ModelCheckpoint(
                        dirpath=ckpt,
                        filename='checkpoint_{epoch}',
                        every_n_epochs=self.hyperparams.get('log_freq_uncertainty', 10),
                        auto_insert_metric_name=False,
                        save_top_k=-1,
                        save_last=False,
                    )
                ]
            else:
                callbacks = []
            trainer = pl.Trainer(
                accelerator=device,
                devices=num_devices,
                max_epochs=epochs,
                logger=logger,
                num_sanity_val_steps=0,
                deterministic=True,
                precision='bf16',
                callbacks=callbacks,
            )
            print(f'---> Training ensemble member {idx}...')
            trainer.fit(bl, datamodule=dm)
            if self.log_wandb:
                wandb.finish()  # necessary for each run to be actually logged
                trainer.logger = None  # avoid error in predict method
                trainer.loggers = []
            self.trainers.append(trainer)

    def predict(
        self,
        dataset: LightningDataModule,
        avg: bool = False,
        ckpt_dir: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict with ensemble."""
        print('---> Computing ensemble prediction...')
        bl_predictions: list = []

        for idx, (tr, bl) in enumerate(zip(self.trainers, self.learners)):
            if ckpt_dir is not None:
                if epoch is None:
                    raise ValueError('Checkpoint path and epoch must be provided.')
                ckpt = os.path.join(
                    ckpt_dir, f'member_{idx}', f'checkpoint_{epoch}.ckpt'
                )
                model = BaseLearner.load_from_checkpoint(ckpt)
                model.eval()
                output = tr.predict(model, datamodule=dataset)
            else:
                output = tr.predict(bl, datamodule=dataset)
            this_prediction = [i[0] for i in output]
            bl_predictions.append(torch.cat(this_prediction, dim=0))
        ensemble_prediction = torch.stack(tuple(bl_predictions))
        if avg:
            ensemble_prediction = torch.mean(ensemble_prediction, dim=0)
        return ensemble_prediction

    def _init_ensemble(self):
        """Initialize NN weights."""
        for idx, bl in enumerate(self.learners):
            bl.set_seed(self.seed + idx)
            bl.init_model_weights()
