import sys
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


def main():
    datasets = ['adult', 'fico', 'higgs', 'magic', 'stumbleupon']
    # for varied_quantity in ['ntrees', 'maxdepth']:
    varied_quantity = 'maxdepth'
    # for scale in ['p', 'logp']:
    for scale in ['logp']:
        for dataset in datasets:
            path = Path(f'results/varyteacher_{varied_quantity}_{dataset}_scale_{scale}_split_none_calibrate_False.json')
            with open(path, 'r') as f:
                data = json.load(f)
            n_trees = data['n_trees']
            max_depth = data['max_depth']
            metric = data['metric']
            scratch_classifier = np.array(data['scratch_classifier'])
            scratch_regressor = np.array(data['scratch_regressor'])
            kd_mse = np.array(data['kd_mse'])
            kd_relerr = np.array(data['kd_relerr'])
            teacher_result = np.array(data['teacher_results'])
            path = Path(f'results/varyteacher_{varied_quantity}_{dataset}_scale_logp_split_crossfit_calibrate_False.json')
            with open(path, 'r') as f:
                data = json.load(f)
            kd_mse_crossfit = np.array(data['kd_mse'])
            # kd_mixed_crossfit = np.array(data['kd_mixed'])
            # kd_relerr_crossfit = np.array(data['kd_relerr'])
            kd_boundfast_crossfit = np.array(data['kd_boundfast'])
            # kd_boundslow_crossfit = np.array(data['kd_boundslow'])
            teacher_result_crossfit = np.array(data['teacher_results'])

            x_var = n_trees if max_depth is None else max_depth
            plt.figure(figsize=(6, 3.5))
            # plt.errorbar(x_var, [np.mean(scratch_classifier)] * len(x_var),
            #              [np.std(scratch_classifier)] * len(x_var),
            #             label='From scratch, classifier')
            # plt.errorbar(x_var, [np.mean(scratch_regressor)] * len(x_var),
            #              [np.std(scratch_regressor)] * len(x_var),
            #             label='From scratch')
            plt.plot(x_var, [np.mean(scratch_regressor)] * len(x_var), 'b', label=r'Student without KD')
            plt.fill_between(x_var, [np.mean(scratch_regressor) - np.std(scratch_regressor)/np.sqrt(scratch_regressor.shape[0])] * len(x_var),
                             [np.mean(scratch_regressor) + np.std(scratch_regressor)/np.sqrt(scratch_regressor.shape[0])] * len(x_var), color='b', alpha=0.1)
            # plt.errorbar(x_var, np.mean(kd_mse, 0), np.std(kd_mse, 0), label='KD, no crossfitting')
            plt.plot(x_var, np.mean(kd_mse, 0), 'c', label='KD Student: 10 trees, max depth=$\infty$')
            plt.fill_between(x_var, np.mean(kd_mse, 0) - np.std(kd_mse, 0)/np.sqrt(kd_mse.shape[0]),
                             np.mean(kd_mse, 0) + np.std(kd_mse, 0)/np.sqrt(kd_mse.shape[0]), color='c', alpha=0.1)
            # plt.errorbar(x_var, np.mean(kd_mse_crossfit, 0), np.std(kd_mse_crossfit, 0),
            #              linestyle='--', label='KD, w/ crossfitting')
            plt.plot(x_var, np.mean(kd_mse_crossfit, 0), 'g', label='Cross-fit KD Student')
            plt.fill_between(x_var, np.mean(kd_mse_crossfit, 0) - np.std(kd_mse_crossfit, 0)/np.sqrt(kd_mse_crossfit.shape[0]),
                             np.mean(kd_mse_crossfit, 0) + np.std(kd_mse_crossfit, 0)/np.sqrt(kd_mse_crossfit.shape[0]), color='g', alpha=0.1)
            # plt.errorbar(x_var, np.mean(kd_mixed_crossfit, 0), np.std(kd_mixed_crossfit, 0),
            #              label='KD, mixed, w/ crossfitting')
            # plt.errorbar(x_var, np.mean(kd_relerr, 0), np.std(kd_relerr, 0),
            #              label='KD, relative error, no crossfitting')
            # plt.errorbar(x_var, np.mean(kd_relerr_crossfit, 0), np.std(kd_relerr_crossfit, 0),
            #              linestyle='--', label='KD, relative error, w/ crossfitting')
            # plt.errorbar(x_var, np.mean(kd_boundfast_crossfit, 0), np.std(kd_boundfast_crossfit, 0),
            #              linestyle='--', label='KD, gamma from fast rate upperbound, w/ crossfitting')
            plt.plot(x_var, np.mean(kd_boundfast_crossfit, 0), 'r', label='Enhanced KD Student')
            plt.fill_between(x_var, np.mean(kd_boundfast_crossfit, 0) - np.std(kd_boundfast_crossfit, 0)/np.sqrt(kd_boundfast_crossfit.shape[0]),
                             np.mean(kd_boundfast_crossfit, 0) + np.std(kd_boundfast_crossfit, 0)/np.sqrt(kd_boundfast_crossfit.shape[0]), color='r', alpha=0.1)
            # plt.errorbar(x_var, np.mean(kd_boundslow_crossfit, 0), np.std(kd_boundslow_crossfit, 0),
            #              linestyle='--', label='KD, gamma from slow rate upperbound, w/ crossfitting')
            # plt.errorbar(x_var, np.mean(teacher_result, 1), np.std(teacher_result, 1),
            #              # linestyle='dotted', label='Teacher, no crossfitting')
            #              linestyle='dotted', label='Teacher')
            plt.plot(x_var, np.mean(teacher_result, 1), 'm--', label='Teacher: 100 trees')
            plt.fill_between(x_var, np.mean(teacher_result, 1) - np.std(teacher_result, 1)/np.sqrt(teacher_result.shape[1]),
                             np.mean(teacher_result, 1) + np.std(teacher_result, 1)/np.sqrt(teacher_result.shape[1]), color='m', alpha=0.1)
            # plt.errorbar(x_var, np.mean(teacher_result_crossfit, 1),
            #              np.std(teacher_result_crossfit, 1),
            #              linestyle='dotted', label='Teacher, w/ crossfitting')
            plt.xticks(x_var)
            plt.ylabel(f'Test {metric}', fontsize=14)
            plt.xlabel("Teacher's number of trees" if varied_quantity == 'ntrees' else "Teacher's max tree depth",
                       fontsize=14)
            plt.legend()
            plot_path = Path(f'results/varyteacher_{varied_quantity}_{dataset}_scale_{scale}_crossfit.pdf')
            plt.savefig(str(plot_path), bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()
