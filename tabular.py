import logging
import json
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import torch
import hydra
from omegaconf import OmegaConf, DictConfig

import datamodules
import metrics


class RandomForestRegressorKD(BaseEstimator):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='p'):
        assert scale in ['p', 'logp']
        self.n_estimators = n_estimators
        self.metric = metric
        self.student = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=-1)
        self.scale = scale

    def fit(self, X, y):
        assert y.ndim == 2
        y, teacher_prob = y[:, 0].astype(int), y[:, 1]
        self.student.fit(X, self.combine_y_teacher_prob(y, teacher_prob))
        return self

    def combine_y_teacher_prob(self, y, teacher_prob):
        return teacher_prob if self.scale == 'p' else np.log(np.maximum(teacher_prob, 1 / 500))

    def predict(self, X):
        return self.student.predict(X) if self.scale == 'p' else np.exp(self.student.predict(X))

    def score(self, X, y):
        if y.ndim == 2:
            y, teacher_prob = y[:, 0].astype(int), y[:, 1]
        return self.metric.score(y, self.predict(X))


class RandomForestRegressorKDMixed(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, alpha=1.0, scale='p'):
        assert scale == 'p'
        super().__init__(n_estimators, metric, scale)
        self.alpha = alpha

    def combine_y_teacher_prob(self, y, teacher_prob):
        return (1 - self.alpha) * y + self.alpha * teacher_prob


class RandomForestRegressorKDOrtho(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='logp'):
        assert scale == 'logp'
        super().__init__(n_estimators, metric, scale)

    def combine_y_teacher_prob(self, y, teacher_prob):
        teacher_prob_clipped = np.maximum(teacher_prob, 1 / 500)
        yp_1 = y / teacher_prob_clipped - 1.0
        return np.log(teacher_prob_clipped) + yp_1


def find_optimal_gamma_sampling(phat, bound_fn, y, max_range=10, alpha=1.0, scale='p'):
    assert scale in ['p', 'logp']
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = np.arange(-max_range, max_range, 0.05)[..., None, None]
    if scale == 'p':
        def objective(p):
            return (gamma * (p - phat) - (p - phat))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
            # return (gamma * (p - phat) - (p - phat))**2 + (1 / alpha - 1 + gamma)**2 * (y - p)**2
    else:
        def objective(p):
            return (gamma * (p - phat) - (np.log(p) - np.log(phat)))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
            # return (gamma * (p - phat) - (np.log(p) - np.log(phat)))**2 + (1 / alpha - 1 + gamma)**2 * (y - p)**2
    bound_l, bound_h = bound_fn(phat)
    p_vals = bound_l + np.linspace(0.0, 1.0, 10)[..., None] * (bound_h - bound_l)
    objs = objective(p_vals)
    max_objs = objs.max(axis=1)
    return gamma[np.argmin(max_objs, axis=0)].reshape(*phat_shape)


# As the objective is just a quadratic function in p, we can solve it analytically instead of by
# sampling p.
def find_optimal_gamma(phat, bound_fn, max_range=10, alpha=1.0):
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = np.arange(-max_range, max_range, 0.05)
    # If there's a tie (np.argmin below), take the gamma with the smallest absolute value
    gamma = gamma[np.argsort(np.abs(gamma))][..., None]
    def objective(p):
        # return (gamma * (p - phat) - (p - phat))**2 + (1 / alpha - 1 + gamma)**2 * p * (1 - p)
        return (1 - 2 * gamma) * p**2 - (2 * phat * (gamma - 1)**2 - gamma**2) * p  + (gamma - 1)**2 * phat**2
    bound_l, bound_h = bound_fn(phat)
    # Solve the quadratic to find optimal p
    optimal_p = (2 * phat * (gamma - 1)**2 - gamma**2) / (2 * (1 - 2 * gamma))
    optimal_p = np.clip(optimal_p, bound_l, bound_h)
    objs = np.stack([objective(bound_l), objective(bound_h), objective(optimal_p)])
    max_objs = objs.max(axis=0)
    return gamma[np.argmin(max_objs, axis=0)].reshape(*phat_shape)


def find_optimal_gamma_bound_slow(phat, y, c, max_range=10, scale='p'):
    assert scale in ['p', 'logp']
    phat_shape = phat.shape
    phat = phat.flatten()
    gamma = np.linspace(-max_range, max_range, 1000)[..., None]
    if scale == 'p':
        # objs = gamma ** 2 * phat * (1 - phat) + c * (gamma - 1) ** 4
        objs = gamma ** 2 * (y - phat) ** 2 + c * (gamma - 1) ** 4
    else:
        # objs = gamma ** 2 * phat * (1 - phat) + c * (gamma - 1 / phat) ** 4
        objs = gamma ** 2 * (y - phat) ** 2 + c * (gamma - 1 / phat) ** 4
    return gamma[np.argmin(objs, axis=0)].reshape(*phat_shape)


class RandomForestRegressorKDRelerr(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='p', c=1.0):
        super().__init__(n_estimators, metric, scale)
        self.c = c

    def combine_y_teacher_prob(self, y, teacher_prob):
        bound_l = 0.0 if self.scale == 'p' else 1e-3
        bound_fn = lambda phat: (np.maximum(phat / (1 + self.c), bound_l),
                            np.minimum(phat * (1 + self.c), 1.0))
        if self.scale == 'p':
            gamma = find_optimal_gamma(teacher_prob, bound_fn)
            return teacher_prob + gamma * (y - teacher_prob)
        else:
            teacher_prob_clipped = np.maximum(teacher_prob, 1 / 500)
            teacher_prob_log = np.log(teacher_prob_clipped)
            gamma = find_optimal_gamma_sampling(teacher_prob_clipped, bound_fn, y, scale='logp')
            return teacher_prob_log + gamma * (y - teacher_prob)


class RandomForestRegressorKDAbserr(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='p', c=1.0):
        super().__init__(n_estimators, metric, scale)
        self.c = c

    def combine_y_teacher_prob(self, y, teacher_prob):
        bound_l = 0.0 if self.scale == 'p' else 1e-3
        bound_fn = lambda phat: (np.maximum(phat - self.c, bound_l),
                            np.minimum(phat + self.c, 1.0))
        if self.scale == 'p':
            gamma = find_optimal_gamma(teacher_prob, bound_fn)
            return teacher_prob + gamma * (y - teacher_prob)
        else:
            teacher_prob_clipped = np.maximum(teacher_prob, 1 / 500)
            teacher_prob_log = np.log(teacher_prob_clipped)
            gamma = find_optimal_gamma_sampling(teacher_prob_clipped, bound_fn, y, scale='logp')
            return teacher_prob_log + gamma * (y - teacher_prob)


class RandomForestRegressorKDPower(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='p', tmax=2.0):
        super().__init__(n_estimators, metric, scale)
        self.tmax = tmax

    def combine_y_teacher_prob(self, y, teacher_prob):
        bound_l = 0.0 if self.scale == 'p' else 1e-3
        bound_fn = lambda phat: (np.maximum(phat, bound_l),
                            np.minimum(phat ** (1 / self.tmax), 1.0))
        if self.scale == 'p':
            gamma = find_optimal_gamma(teacher_prob, bound_fn)
            return teacher_prob + gamma * (y - teacher_prob)
        else:
            teacher_prob_clipped = np.maximum(teacher_prob, 1 / 500)
            teacher_prob_log = np.log(teacher_prob_clipped)
            gamma = find_optimal_gamma_sampling(teacher_prob_clipped, bound_fn, y, scale='logp')
            return teacher_prob_log + gamma * (y - teacher_prob)


class RandomForestRegressorKDBoundFast(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='p', c=1.0):
        super().__init__(n_estimators, metric, scale)
        self.c = c

    def combine_y_teacher_prob(self, y, teacher_prob):
        if self.scale == 'p':
            # gamma = self.c / (self.c + teacher_prob * (1 - teacher_prob))
            gamma = self.c / (self.c + (y - teacher_prob)**2)
            return teacher_prob + gamma * (y - teacher_prob)
        else:
            teacher_prob_clipped = np.maximum(teacher_prob, 1 / 500)
            teacher_prob_log = np.log(teacher_prob_clipped)
            # gamma = self.c / teacher_prob_clipped / (self.c + teacher_prob * (1 - teacher_prob))
            gamma = self.c / teacher_prob_clipped / (self.c + (y - teacher_prob)**2)
            return teacher_prob_log + gamma * (y - teacher_prob)


class RandomForestRegressorKDBoundSlow(RandomForestRegressorKD):

    def __init__(self, n_estimators=100, metric=metrics.Accuracy, scale='p', c=1.0):
        super().__init__(n_estimators, metric, scale)
        self.c = c

    def combine_y_teacher_prob(self, y, teacher_prob):
        if self.scale == 'p':
            gamma = find_optimal_gamma_bound_slow(teacher_prob, y, self.c, max_range=100, scale=self.scale)
            return teacher_prob + gamma * (y - teacher_prob)
        else:
            teacher_prob_clipped = np.maximum(teacher_prob, 1 / 500)
            teacher_prob_log = np.log(teacher_prob_clipped)
            gamma = find_optimal_gamma_bound_slow(teacher_prob_clipped, y, self.c, max_range=100, scale=self.scale)
            return teacher_prob_log + gamma * (y - teacher_prob)


def train(model, data, metric=metrics.Accuracy):
    X_train, y_train, X_test, y_test = data
    model.fit(X_train, y_train)
    y_pred = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, 'predict_proba') else model.predict(X_test))
    return metric.score(y_test, y_pred)


def cv(model, data, param_grid):
    X_train, y_train, X_test, y_test = data
    search = GridSearchCV(model, param_grid, refit=False, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_


def train_kd(cls, cfg, data, param_grid, metric=metrics.Accuracy):
    # n_jobs=1 since GridSearchCV already uses n_jobs=-1
    best_params = Parallel(n_jobs=1)(
        delayed(cv)(cls(metric=metric, scale=cfg.scale, n_estimators=n_tree), data, param_grid)
        for n_tree in cfg.n_trees
    )
    results = Parallel(n_jobs=-1)(
        delayed(train)(cls(metric=metric, scale=cfg.scale, n_estimators=n_tree, **best_param),
                       data, metric=metric)
        for n_tree, best_param in zip(list(cfg.n_trees) * cfg.n_repeats, best_params * cfg.n_repeats)
    )
    return results, best_params


# # For interactive use
# cfg = DictConfig({
#     'dataset': DictConfig({'_target_': 'datamodules.MagicTelescope', 'seed': '${seed}'}),
#     # 'dataset': DictConfig({'_target_': 'datamodules.FICO', 'data_dir': '/dfs/scratch0/trid/data/fico/', 'seed': '${seed}'}),
#     'metric': DictConfig({'_target_': 'metrics.AUC'}),
#     'n_trees': [1, 2, 3],
#     'n_repeats': 5,
#     'n_splits': 5,
#     'split_mode': 'crossfit',
#     'calibrate_teacher': False,
#     'scale': 'logp',
#     'seed': 2357
# })


@hydra.main(config_path="cfg", config_name="tabular.yaml")
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
    if cfg.split_mode == 'crossfit':
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        teachers = []
        teacher_test_index = np.empty(len(X_train), dtype=int)
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            X_train_this_split, y_train_this_split = X_train[train_index], y_train[train_index]
            teacher_test_index[test_index] = i
            teacher = teacher_cls(n_estimators=500, n_jobs=-1)
            teacher.fit(X_train_this_split, y_train_this_split)
            teachers.append(teacher)
        teacher_result = np.array([metric.score(y_test, teacher.predict_proba(X_test)[:, 1])
                                   for teacher in teachers])
        teacher_prob = np.stack([teacher.predict_proba(X_train)[:, 1] for teacher in teachers])
        teacher_prob = teacher_prob[teacher_test_index, np.arange(len(X_train))]
    else:
        if cfg.split_mode == 'split':
            X_train, X_train_teacher, y_train, y_train_teacher = train_test_split(
                X_train, y_train, test_size=0.5, random_state=cfg.seed
            )
        else:  # cfg.split_mode == 'none'
            X_train_teacher, y_train_teacher = X_train, y_train
        teacher_result = [
            train(teacher_cls(n_estimators=500, n_jobs=-1),
                  (X_train_teacher, y_train_teacher, X_test, y_test), metric=metric)
            for _ in range(cfg.n_repeats)
        ]
        teacher_result = np.array(teacher_result)
        teacher = teacher_cls(n_estimators=500, n_jobs=-1)
        teacher.fit(X_train_teacher, y_train_teacher)
        teacher_prob = teacher.predict_proba(X_train)[:, 1]

    ##################################### Train students #########################################
    logger.info('Training classifier from scratch')
    scratch_classifier = Parallel(n_jobs=-1)(
        delayed(train)(RandomForestClassifier(n_estimators=n_tree),
                       (X_train, y_train, X_test, y_test), metric=metric)
        for n_tree in list(cfg.n_trees) * cfg.n_repeats
    )
    scratch_classifier = np.array(scratch_classifier).reshape(cfg.n_repeats, -1)
    logger.info('Training regressor from scratch')
    scratch_regressor = Parallel(n_jobs=-1)(
        delayed(train)(RandomForestRegressor(n_estimators=n_tree),
                       (X_train, y_train, X_test, y_test), metric=metric)
        for n_tree in list(cfg.n_trees) * cfg.n_repeats
    )
    scratch_regressor = np.array(scratch_regressor).reshape(cfg.n_repeats, -1)

    logger.info('Training student with knowledge distillation')
    y_train_w_teacherprob = np.stack([y_train, teacher_prob], axis=1)
    kd_mse = Parallel(n_jobs=-1)(
        delayed(train)(RandomForestRegressorKD(metric=metric, scale=cfg.scale, n_estimators=n_tree),
                       (X_train, y_train_w_teacherprob, X_test, y_test), metric=metric)
        for n_tree in list(cfg.n_trees) * cfg.n_repeats
    )
    kd_mse = np.array(kd_mse).reshape(cfg.n_repeats, -1)

    logger.info('Training student with knowledge distillation upper bound fast rate')
    c_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
    kd_boundfast, kd_boundfast_params = train_kd(RandomForestRegressorKDBoundFast, cfg,
                                                (X_train, y_train_w_teacherprob, X_test, y_test),
                                                 {'c': c_values}, metric=metric)
    kd_boundfast = np.array(kd_boundfast).reshape(cfg.n_repeats, -1)

    logger.info('Training student with knowledge distillation upper bound slow rate')
    c_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
    kd_boundslow, kd_boundslow_params = train_kd(RandomForestRegressorKDBoundSlow, cfg,
                                                (X_train, y_train_w_teacherprob, X_test, y_test),
                                                 {'c': c_values}, metric=metric)
    kd_boundslow = np.array(kd_boundslow).reshape(cfg.n_repeats, -1)

    if cfg.scale == 'p':
        logger.info('Training student with mixed knowledge distillation')
        alpha_values = np.linspace(0.0, 1.0, 11)
        kd_mixed, kd_mixed_params = train_kd(RandomForestRegressorKDMixed, cfg,
                                            (X_train, y_train_w_teacherprob, X_test, y_test),
                                            {'alpha': alpha_values}, metric=metric)
        kd_mixed = np.array(kd_mixed).reshape(cfg.n_repeats, -1)

    logger.info('Training student with relerr knowledge distillation')
    c_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    kd_relerr, kd_relerr_params = train_kd(RandomForestRegressorKDRelerr, cfg,
                                          (X_train, y_train_w_teacherprob, X_test, y_test),
                                           {'c': c_values}, metric=metric)
    kd_relerr = np.array(kd_relerr).reshape(cfg.n_repeats, -1)

    logger.info('Training student with abserr knowledge distillation')
    c_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    kd_abserr, kd_abserr_params = train_kd(RandomForestRegressorKDAbserr, cfg,
                                          (X_train, y_train_w_teacherprob, X_test, y_test),
                                           {'c': c_values}, metric=metric)
    kd_abserr = np.array(kd_abserr).reshape(cfg.n_repeats, -1)

    logger.info('Training student with power knowledge distillation')
    tmax_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    kd_power, kd_power_params = train_kd(RandomForestRegressorKDPower, cfg,
                                         (X_train, y_train_w_teacherprob, X_test, y_test),
                                         {'tmax': tmax_values}, metric=metric)
    kd_power = np.array(kd_power).reshape(cfg.n_repeats, -1)

    ##################################### Dump to JSON file #######################################
    logger.info('Saving results to disk')
    file_name = f'tabular_{datamodule.name}_scale_{cfg.scale}_split_{cfg.split_mode}_calibrate_{cfg.calibrate_teacher}.json'
    data = {
        'n_trees': list(cfg.n_trees),
        'metric': type(metric).__name__,
        'scratch_classifier': scratch_classifier.tolist(),
        'scratch_regressor': scratch_regressor.tolist(),
        'kd_mse': kd_mse.tolist(),
        'kd_boundfast': kd_boundfast.tolist(),
        'kd_boundfast_params': kd_boundfast_params,
        'kd_boundslow': kd_boundslow.tolist(),
        'kd_boundslow_params': kd_boundslow_params,
        'kd_mixed': kd_mixed.tolist() if cfg.scale == 'p' else None,
        'kd_mixed_params': kd_mixed_params if cfg.scale == 'p' else None,
        'kd_relerr': kd_relerr.tolist(),
        'kd_relerr_params': kd_relerr_params,
        'kd_abserr': kd_abserr.tolist(),
        'kd_abserr_params': kd_abserr_params,
        'kd_power': kd_power.tolist(),
        'kd_power_params': kd_power_params,
        'teacher_result': teacher_result.tolist(),
    }
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    # Also write the same content to `results` directory
    path = Path(hydra.utils.get_original_cwd()) / 'results' / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f'Saved results to {str(path)}')

    # # plt.figure()
    # # plt.hist(teacher.predict_proba(X_train_teacher)[:, 1], bins=10)
    # # plt.savefig(f'histogram_p_hat_{dataset_name}_split_True_Xtrainteacher.pdf', bbox_inches='tight')
    # # plt.close()

    # # plt.figure()
    # # plt.hist(teacher_prob, bins=10)
    # # plt.savefig(f'histogram_p_hat_{dataset_name}_split_True.pdf', bbox_inches='tight')
    # # plt.close()


if __name__ == '__main__':
    main()
