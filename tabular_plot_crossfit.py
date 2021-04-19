import sys
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate tabular crossfit figures')
    parser.add_argument('-d', '--datasets',
                        nargs='+', 
                        dest='datasets',
                        default=['adult', 'fico', 'higgs', 'magic', 'stumbleupon'],
                        help='Produce plots for these datasets')
    parser.add_argument('-m', '--methods',
                        nargs='+',
                        dest='methods',
                        default=['teacher','scratch','vanilla','cf','enhanced'],
                        help='Produce plots for these methods')
    args = parser.parse_args()

    datasets = args.datasets
    # Convert methods to a set and test whether particular methods are present
    methods = set(args.methods)
    plot_teacher = 'teacher' in methods
    plot_scratch = 'scratch' in methods
    plot_vanilla = 'vanilla' in methods
    plot_cf = 'cf' in methods
    plot_enhanced = 'enhanced' in methods
    for dataset in datasets:
        path = Path(f'results/tabular_{dataset}_scale_logp_split_none_calibrate_False.json')
        with open(path, 'r') as f:
            data = json.load(f)
        n_trees = data['n_trees']
        metric = data['metric']
        scratch_classifier = np.array(data['scratch_classifier'])
        scratch_regressor = np.array(data['scratch_regressor'])
        kd_mse = np.array(data['kd_mse'])
        teacher_result = np.array(data['teacher_result'])
        path = Path(f'results/tabular_{dataset}_scale_p_split_crossfit_calibrate_False.json')
        with open(path, 'r') as f:
            data = json.load(f)
        kd_mse_crossfit = np.array(data['kd_mse'])
        kd_relerr = np.array(data['kd_relerr'])
        kd_boundfast = np.array(data['kd_boundfast'])
        teacher_result_crossfit = np.array(data['teacher_result'])

        plt.figure(figsize=(6, 3.5))
        # plt.errorbar(n_trees, np.mean(scratch_classifier, 0), np.std(scratch_classifier, 0),
        #             label='From scratch, classifier')
        # plt.errorbar(n_trees, np.mean(scratch_regressor, 0), np.std(scratch_regressor, 0),
        
        # Initially plot scratch to ensure axes have a consistent size
        # Then remove line if not requested in the final plot
        scratch_label = 'Student without KD'
        scratch_color = 'b'
        myplot = plt.plot(n_trees, np.mean(scratch_regressor, 0), scratch_color, label=scratch_label)
        if plot_scratch:
            print(f'{scratch_label} {np.mean(scratch_regressor, 0)}')
            std_error = np.std(scratch_regressor, 0)/np.sqrt(scratch_regressor.shape[0])
            plt.fill_between(n_trees, np.mean(scratch_regressor, 0) - std_error,
                             np.mean(scratch_regressor, 0) + std_error, color=scratch_color, alpha=0.1)
        else:
            line = myplot.pop(0)
            line.remove()
            
        # plt.errorbar(n_trees, np.mean(kd_mse, 0), np.std(kd_mse, 0), label='KD, no cross-fitting')
        if plot_vanilla:
            vanilla_label = 'KD Student'
            print(f'{vanilla_label} {np.mean(kd_mse, 0)}')
            plt.plot(n_trees, np.mean(kd_mse, 0), 'c', label=vanilla_label)
            std_error = np.std(kd_mse, 0)/np.sqrt(kd_mse.shape[0])
            plt.fill_between(n_trees, np.mean(kd_mse, 0) - std_error,
                             np.mean(kd_mse, 0) + std_error, color='c', alpha=0.1)
        # plt.errorbar(n_trees, np.mean(kd_mse_crossfit, 0), np.std(kd_mse_crossfit, 0), label='KD, w/ cross-fitting')
        if plot_cf:
            cf_label = 'Cross-fit KD Student'
            print(f'{cf_label} {np.mean(kd_mse_crossfit, 0)}')
            plt.plot(n_trees, np.mean(kd_mse_crossfit, 0), 'g', label=cf_label)
            std_error = np.std(kd_mse_crossfit, 0)/np.sqrt(kd_mse_crossfit.shape[0])
            plt.fill_between(n_trees, np.mean(kd_mse_crossfit, 0) - std_error,
                             np.mean(kd_mse_crossfit, 0) + std_error, color='g', alpha=0.1)
        # plt.errorbar(n_trees, np.mean(kd_relerr, 0), np.std(kd_relerr, 0), label='KD, relative error, w/ crossfitting')
        # plt.errorbar(n_trees, np.mean(kd_boundfast, 0), np.std(kd_boundfast, 0), label=r'$\gamma$-corrected KD, w/ crossfitting')
        if plot_enhanced:
            enhanced_label = 'Enhanced KD Student'
            print(f'{enhanced_label} {np.mean(kd_boundfast, 0)}')
            enhanced_color = 'r'
            plt.plot(n_trees, np.mean(kd_boundfast, 0), enhanced_color, label=enhanced_label)
            std_error = np.std(kd_boundfast, 0)/np.sqrt(kd_boundfast.shape[0])
            plt.fill_between(n_trees, np.mean(kd_boundfast, 0) - std_error,
                             np.mean(kd_boundfast, 0) + std_error, color=enhanced_color, alpha=0.1)
        # plt.errorbar(n_trees, [np.mean(teacher_result)] * len(n_trees),
        #             [np.std(teacher_result)] * len(n_trees), linestyle='--', label='Teacher (500 trees)')
        if plot_teacher:
            teacher_label = 'Teacher: 500 trees'
            #if dataset == 'adult':
            #    teacher_label += f', {round(np.mean(teacher_result),3)} {metric}'
            print(f'{teacher_label} {np.mean(teacher_result)}')
            plt.plot(n_trees, [np.mean(teacher_result)] * len(n_trees), 'm--', label=teacher_label)
            std_error = np.std(teacher_result)/np.sqrt(teacher_result.shape[0])
            plt.fill_between(n_trees, [np.mean(teacher_result) - std_error] * len(n_trees),
                             [np.mean(teacher_result) + std_error] * len(n_trees), color='m', alpha=0.1)
        # plt.errorbar(n_trees, [np.mean(teacher_result_crossfit)] * len(n_trees),
        #             [np.std(teacher_result_crossfit)] * len(n_trees), linestyle='--', label='Teacher, w/ crossfitting')
        plt.xticks(n_trees)
        plt.ylabel(f'Test {metric}', fontsize=14)
        plt.xlabel("Student's number of trees", fontsize=14)
        plt.legend(loc='lower right')
        plot_path = Path(f'results/tabular_{dataset}_crossfit'
                         f'-teacher{plot_teacher}_scratch{plot_scratch}_vanilla{plot_vanilla}_cf{plot_cf}_enhanced{plot_enhanced}.pdf')
        plt.savefig(str(plot_path), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
