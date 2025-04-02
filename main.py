"""Perform experiments."""

import os
import re
from dataclasses import asdict
from datetime import datetime
from typing import Final

import click
import torch
from torch.nn.utils import parameters_to_vector

from shortcuts.config import (
    BaseLearnerConfig,
    ExperimentConfig,
    PathConfig,
)
from shortcuts.datamodules import DataModuleFactory
from shortcuts.ensemble import BaseLearner, EnsembleLearner
from shortcuts.eval import Evaluator
from shortcuts.models import ModelFactory

from root_path import ROOT

EXPERIMENTS: Final[list[str]] = [
    'mnist3',
    'mnist3_ood',
    'cmnist3',
    'cmnist3_ood',
    'pmnist3',
    'pmnist3_ood',
]
SHORTCUT_STRENGTHS: Final[list[float]] = [0.5, 0.75, 0.95, 1.0]
PATHS: Final[PathConfig] = PathConfig(
    os.path.join(ROOT, 'configs/'),
    os.path.join(ROOT, 'datasets/'),
    os.path.join(ROOT, 'checkpoints/'),
    os.path.join(ROOT, 'results/'),
)


@click.command()
@click.option('-expid', '--experiment_id', required=True)
@click.option('-rs', '--random_seed', required=True)
def main(experiment_id: str, random_seed: int) -> None:
    """Run experiments."""
    # load configuration
    if 'mnist3' in experiment_id:
        config_file_path = os.path.join(PATHS.config, 'config_mnist3.yml')
    else:
        raise NotImplementedError(f'Experiment `{experiment_id}` not implemented.')
    cfg = ExperimentConfig(config_file_path, experiment_id, int(random_seed))

    num_classes = cfg.config_data['num_classes']
    cfg_model = cfg.config_model
    cfg_training = cfg.config_training
    model_args: dict = cfg_model.get('model_specs', {})
    hyperparams = cfg_training['hyperparams']
    log_wandb = cfg_training['log']

    match experiment_id:
        case 'mnist3':
            shortcut_strengths = [1.0]
            num_channels = 1
            dataset_name = 'mnist3'
            dataset_name_test = None
        case 'mnist3_ood':
            shortcut_strengths = [1.0]
            num_channels = 1
            dataset_name = 'mnist3'
            dataset_name_test = 'mnist0'
        case 'cmnist3':
            shortcut_strengths = SHORTCUT_STRENGTHS
            num_channels = 3
            dataset_name = 'cmnist3'
            dataset_name_test = 'mnist3c'
        case 'cmnist3_ood':
            shortcut_strengths = SHORTCUT_STRENGTHS
            num_channels = 3
            dataset_name = 'cmnist3'
            dataset_name_test = 'mnist0'
        case 'pmnist3':
            shortcut_strengths = SHORTCUT_STRENGTHS
            num_channels = 3
            dataset_name = 'pmnist3'
            dataset_name_test = 'mnist3'
        case 'pmnist3_ood':
            shortcut_strengths = SHORTCUT_STRENGTHS
            num_channels = 3
            dataset_name = 'pmnist3'
            dataset_name_test = 'mnist0'
        case _:
            raise NotImplementedError(f'Experiment `{experiment_id}` not implemented.')

    for s in shortcut_strengths:

        time_in = datetime.now()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # fetch data
        data_specs = cfg.get_data_specs()
        data_specs.update(
            {
                'dataset_name': dataset_name,
                'save_to': os.path.join(PATHS.data, dataset_name),
                'shortcut_strength': s,
                'num_channels': num_channels,
            }
        )
        if dataset_name == 'pmnist3':
            data_specs.update({'patch': True, 'patch_prob': s})
        dataset = DataModuleFactory.create_module(**data_specs)

        # define learner
        learners: list = []
        for _ in range(cfg_model['ensemble_size']):
            model = ModelFactory.create_model(
                model_name=cfg_model['model_name'],
                num_classes=num_classes,
                num_channels=num_channels,
                **model_args,
            )
            base_learner_config = BaseLearnerConfig(
                device='cpu',
                model=model,
                num_classes=num_classes,
                log_wandb=log_wandb,
                hyperparams=hyperparams,
            )
            learners.append(BaseLearner(**asdict(base_learner_config)))

        # create checkpoint dir
        ckpt = os.path.join(PATHS.checkpoints, ts)
        os.makedirs(ckpt, exist_ok=True)

        # build ensemble
        ensemble = EnsembleLearner(
            ensemble_id=experiment_id,
            device='cpu',
            learners=learners,
            hyperparams=hyperparams,
            log_wandb=log_wandb,
            seed=cfg.seed,
            ckpt=ckpt,
        )
        param_vecs: list = []
        for learner in ensemble.learners:
            param_vecs.append(parameters_to_vector(learner.parameters()))
        if any([torch.equal(x, param_vecs[0]) for x in param_vecs[1:]]):
            raise ValueError('Initial parameters of ensemble members are the same.')

        # train
        print(f'-----> Start training for experiment {experiment_id}')
        ensemble.train(
            dm=dataset,
            epochs=cfg_training['num_epochs'],
            device='cpu',
            num_devices=1,
        )

        # evaluate
        this_result_dir = os.path.join(PATHS.results, experiment_id, f'{ts}')
        os.makedirs(this_result_dir, exist_ok=True)
        ckpts = [i for i in os.listdir(os.path.join(ckpt, 'member_0')) if '.ckpt' in i]
        epochs = sum([re.findall(r'checkpoint_(\d+)\.ckpt', i) for i in ckpts], [])

        if dataset_name_test is not None:
            data_specs.update(
                {
                    'dataset_name': dataset_name_test,
                    'save_to': os.path.join(PATHS.data, dataset_name_test),
                    'num_channels': num_channels,
                }
            )
            dataset = DataModuleFactory.create_module(**data_specs)

        evaluator_full = Evaluator(
            ensemble=ensemble, dataset=dataset, num_classes=num_classes
        )
        evaluator_top20 = Evaluator(
            ensemble=ensemble,
            dataset=dataset,
            num_classes=num_classes,
            top_u_percentile=0.2,
        )
        evaluators = [evaluator_full, evaluator_top20]
        suffices = ['full', 'top20']
        for e, sfx in zip(evaluators, suffices):
            for i in epochs:
                e.prepare_predictions(ckpt_dir=ckpt, epoch=int(i))
                e.evaluate_all(epoch=int(i))  # per checkpoint
            e.prepare_predictions()  # final model
            e.evaluate_all()
            e.save_metric_dict(os.path.join(this_result_dir, f'metrics_{sfx}.json'))
            e.save_uncertainty_dict(
                os.path.join(this_result_dir, f'uncertainty_{sfx}.json')
            )
            e.save_num_obs_dict(os.path.join(this_result_dir, f'num_obs_{sfx}.json'))
            e.save_uncertainty_binned(
                os.path.join(this_result_dir, f'uncertainty_binned_{sfx}.csv')
            )

        # log
        time_elapsed = datetime.now() - time_in
        cfg.config.update({'time_elapsed': time_elapsed.seconds})
        cfg.config.update({'inherent_shortcut_strength': s})
        cfg.to_yaml(ts, os.path.join(this_result_dir, 'config_out.yml'))


if __name__ == '__main__':
    main()
