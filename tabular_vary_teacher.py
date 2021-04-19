import logging
import json
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, KFold

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import torch
import hydra
from omegaconf import OmegaConf, DictConfig

import datamodules
import metrics

from tabular import RandomForestRegressorKD, RandomForestRegressorKDMixed
from tabular import RandomForestRegressorKDOrtho
from tabular import RandomForestRegressorKDBoundFast, RandomForestRegressorKDBoundSlow
from tabular import RandomForestRegressorKDRelerr, RandomForestRegressorKDAbserr
from tabular import RandomForestRegressorKDPower
from tabular import train, cv


def train_kd_vary_teacher(cls, cfg, data_list, param_grid, metric=metrics.Accuracy):
    # n_jobs=1 since GridSearchCV already uses n_jobs=-1
    best_params = Parallel(n_jobs=1)(
        delayed(cv)(cls(metric=metric, scale=cfg.scale, n_estimators=cfg.n_tree_student),
                    data, param_grid)
        for data in data_list
    )
    results = Parallel(n_jobs=-1)(
        delayed(train)(cls(metric=metric, scale=cfg.scale, n_estimators=cfg.n_tree_student,
                           **best_param), data, metric=metric)
        for data, best_param in zip(data_list * cfg.n_repeats, best_params * cfg.n_repeats)
    )
    return results, best_params


# For interactive use
cfg = DictConfig({
    'dataset': DictConfig({'_target_': 'datamodules.MagicTelescope', 'seed': '${seed}'}),
    # 'dataset': DictConfig({'_target_': 'datamodules.FICO', 'data_dir': '/dfs/scratch0/trid/data/fico/', 'seed': '${seed}'}),
    'metric': DictConfig({'_target_': 'metrics.AUC'}),
    'n_trees': [10, 20, 30],
    'n_repeats': 5,
    'n_splits': 5,
    'split_mode': 'crossfit',
    'calibrate_teacher': False,
    'scale': 'logp',
    'seed': 2357
})


@hydra.main(config_path="cfg", config_name="tabular_vary_teacher.yaml")
def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    #################################### Load Data ##########################################
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup()
    X_train, y_train = datamodule.X_train, datamodule.y_train
    X_test, y_test = datamodule.X_test, datamodule.y_test

    metric = hydra.utils.instantiate(cfg.metric)

    #################################### Train Teacher #######################################
    logger.info('Training teacher')
    if not cfg.calibrate_teacher:
        teacher_cls = RandomForestClassifier
    else:
        teacher_cls = lambda *args, **kwargs: CalibratedClassifierCV(RandomForestClassifier(*args,
                                                                                       **kwargs))

    if cfg.max_depth is None:
        kwargs_list = [{'n_estimators': n_tree} for n_tree in cfg.n_trees]
        varied_quantity = 'ntrees'
    else:
        kwargs_list = [{'max_depth': d} for d in cfg.max_depth]
        varied_quantity = 'maxdepth'
    teacher_probs = []
    teacher_results = []
    if cfg.split_mode == 'crossfit':
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        for kwargs in kwargs_list:
            teachers = []
            teacher_test_index = np.empty(len(X_train), dtype=int)
            for i, (train_index, test_index) in enumerate(kf.split(X_train)):
                X_train_this_split, y_train_this_split = X_train[train_index], y_train[train_index]
                teacher_test_index[test_index] = i
                teacher = teacher_cls(**kwargs, n_jobs=-1)
                teacher.fit(X_train_this_split, y_train_this_split)
                teachers.append(teacher)
            teacher_result = np.array([metric.score(y_test, teacher.predict_proba(X_test)[:, 1])
                                    for teacher in teachers])
            teacher_prob = np.stack([teacher.predict_proba(X_train)[:, 1] for teacher in teachers])
            teacher_prob = teacher_prob[teacher_test_index, np.arange(len(X_train))]
            teacher_probs.append(teacher_prob)
            teacher_results.append(teacher_result)
    else:
        if cfg.split_mode == 'split':
            X_train, X_train_teacher, y_train, y_train_teacher = train_test_split(
                X_train, y_train, test_size=0.5, random_state=cfg.seed
            )
        else:  # cfg.split_mode == 'none'
            X_train_teacher, y_train_teacher = X_train, y_train
        for kwargs in kwargs_list:
            teacher_result = [
                train(teacher_cls(**kwargs, n_jobs=-1),
                    (X_train_teacher, y_train_teacher, X_test, y_test), metric=metric)
                for _ in range(cfg.n_repeats)
            ]
            teacher_result = np.array(teacher_result)
            teacher = teacher_cls(**kwargs, n_jobs=-1)
            teacher.fit(X_train_teacher, y_train_teacher)
            teacher_prob = teacher.predict_proba(X_train)[:, 1]
            teacher_probs.append(teacher_prob)
            teacher_results.append(teacher_result)
    teacher_results = np.array(teacher_results)

    ##################################### Train students #########################################
    logger.info('Training classifier from scratch')
    scratch_classifier = Parallel(n_jobs=-1)(
        delayed(train)(RandomForestClassifier(n_estimators=cfg.n_tree_student),
                       (X_train, y_train, X_test, y_test),
                       metric=metric) for _ in range(cfg.n_repeats)
    )
    scratch_classifier = np.array(scratch_classifier).reshape(cfg.n_repeats, -1)
    logger.info('Training regressor from scratch')
    scratch_regressor = Parallel(n_jobs=-1)(
        delayed(train)(RandomForestRegressor(n_estimators=cfg.n_tree_student),
                       (X_train, y_train, X_test, y_test),
                       metric=metric) for _ in range(cfg.n_repeats)
    )
    scratch_regressor = np.array(scratch_regressor).reshape(cfg.n_repeats, -1)

    y_train_w_teacherprobs = [np.stack([y_train, teacher_prob], axis=1)
                              for teacher_prob in teacher_probs]
    logger.info('Training student with knowledge distillation')
    kd_mse = Parallel(n_jobs=-1)(
        delayed(train)(RandomForestRegressorKD(metric=metric, scale=cfg.scale,
                                               n_estimators=cfg.n_tree_student),
                       (X_train, y_train_w_teacherprob, X_test, y_test), metric=metric)
        for y_train_w_teacherprob in y_train_w_teacherprobs * cfg.n_repeats
    )
    kd_mse = np.array(kd_mse).reshape(cfg.n_repeats, -1)

    if cfg.scale == 'p':
        logger.info('Training student with mixed knowledge distillation')
        alpha_values = np.linspace(0.0, 1.0, 11)
        kd_mixed, kd_mixed_params = train_kd_vary_teacher(
            RandomForestRegressorKDMixed, cfg,
            [(X_train, y_train_w_teacherprob, X_test, y_test) for y_train_w_teacherprob in y_train_w_teacherprobs],
            {'alpha': alpha_values}, metric=metric
        )
        kd_mixed = np.array(kd_mixed).reshape(cfg.n_repeats, -1)

    if cfg.scale == 'logp':
        logger.info('Training student with ortholoss knowledge distillation')
        kd_ortholoss = Parallel(n_jobs=-1)(
            delayed(train)(RandomForestRegressorKDOrtho(metric=metric, scale=cfg.scale,
                                                        n_estimators=cfg.n_tree_student),
                           (X_train, y_train_w_teacherprob, X_test, y_test), metric=metric)
            for y_train_w_teacherprob in y_train_w_teacherprobs * cfg.n_repeats
        )
        kd_ortholoss = np.array(kd_ortholoss).reshape(cfg.n_repeats, -1)

    logger.info('Training student with knowledge distillation upper bound fast rate')
    c_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
    kd_boundfast, kd_boundfast_params = train_kd_vary_teacher(
        RandomForestRegressorKDBoundFast, cfg,
        [(X_train, y_train_w_teacherprob, X_test, y_test) for y_train_w_teacherprob in y_train_w_teacherprobs],
        {'c': c_values}, metric=metric
    )
    kd_boundfast = np.array(kd_boundfast).reshape(cfg.n_repeats, -1)

    logger.info('Training student with knowledge distillation upper bound slow rate')
    c_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
    kd_boundslow, kd_boundslow_params = train_kd_vary_teacher(
        RandomForestRegressorKDBoundSlow, cfg,
        [(X_train, y_train_w_teacherprob, X_test, y_test) for y_train_w_teacherprob in y_train_w_teacherprobs],
        {'c': c_values}, metric=metric
    )
    kd_boundslow = np.array(kd_boundslow).reshape(cfg.n_repeats, -1)

    logger.info('Training student with relerr knowledge distillation')
    c_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    kd_relerr, kd_relerr_params = train_kd_vary_teacher(
        RandomForestRegressorKDRelerr, cfg,
        [(X_train, y_train_w_teacherprob, X_test, y_test) for y_train_w_teacherprob in y_train_w_teacherprobs],
        {'c': c_values}, metric=metric
    )
    kd_relerr = np.array(kd_relerr).reshape(cfg.n_repeats, -1)

    logger.info('Training student with abserr knowledge distillation')
    c_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    kd_abserr, kd_abserr_params = train_kd_vary_teacher(
        RandomForestRegressorKDAbserr, cfg,
        [(X_train, y_train_w_teacherprob, X_test, y_test) for y_train_w_teacherprob in y_train_w_teacherprobs],
        {'c': c_values}, metric=metric
    )
    kd_abserr = np.array(kd_abserr).reshape(cfg.n_repeats, -1)

    logger.info('Training student with power knowledge distillation')
    tmax_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    kd_power, kd_power_params = train_kd_vary_teacher(
        RandomForestRegressorKDPower, cfg,
        [(X_train, y_train_w_teacherprob, X_test, y_test) for y_train_w_teacherprob in y_train_w_teacherprobs],
        {'tmax': tmax_values}, metric=metric
    )
    kd_power = np.array(kd_power).reshape(cfg.n_repeats, -1)

    ##################################### Dump to JSON file #######################################
    logger.info('Saving results to disk')
    file_name = f'varyteacher_{varied_quantity}_{datamodule.name}_scale_{cfg.scale}_split_{cfg.split_mode}_calibrate_{cfg.calibrate_teacher}.json'
    data = {
        'n_trees': list(cfg.n_trees),
        'max_depth': list(cfg.max_depth) if cfg.max_depth is not None else None,
        'metric': type(metric).__name__,
        'scratch_classifier': scratch_classifier.tolist(),
        'scratch_regressor': scratch_regressor.tolist(),
        'kd_mse': kd_mse.tolist(),
        'kd_mixed': kd_mixed.tolist() if cfg.scale == 'p' else None,
        'kd_mixed_params': kd_mixed_params if cfg.scale == 'p' else None,
        'kd_ortholoss': kd_ortholoss.tolist() if cfg.scale == 'logp' else None,
        'kd_boundfast': kd_boundfast.tolist(),
        'kd_boundfast_params': kd_boundfast_params,
        'kd_boundslow': kd_boundslow.tolist(),
        'kd_boundslow_params': kd_boundslow_params,
        'kd_relerr': kd_relerr.tolist(),
        'kd_relerr_params': kd_relerr_params,
        'kd_abserr': kd_abserr.tolist(),
        'kd_abserr_params': kd_abserr_params,
        'kd_power': kd_power.tolist(),
        'kd_power_params': kd_power_params,
        'teacher_results': teacher_results.tolist(),
    }
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    # Also write the same content to `results` directory
    path = Path(hydra.utils.get_original_cwd()) / 'results' / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f'Saved results to {str(path)}')


if __name__ == '__main__':
    main()
