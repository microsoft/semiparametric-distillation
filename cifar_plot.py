import sys
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


def main():
    start_idx = 1
    depth = [8, 14, 20, 32, 44, 56]
    kd_acc = np.array([(81.4, 0.1), (84.57, 0.1), (83.67, 0.1), (82.73, 0.7), (82.5, 0.3), (81.65, 0.4)])
    kd_crossfit_acc = np.array([(79.52, 0.3), (83.44, 0.2), (83.93, 0.3), (83.67, 0.1), (83.51, 0.5), (83.23, 0.6)])
    kd_crossfit_gamma_acc = np.array([(0.8085, 0.2), (83.93, 0.2), (84.61, 0.3), (83.98, 0.4), (83.93, 0.3), (83.52, 0.4)])
    teacher_acc = np.array([(77.82, 0.4), (85.03, 0.3), (87.15, 0.4), (86.45, 0.5), (87.00, 0.7), (87.90, 0.9)])
    teacher_crossfit_acc = np.array([(76.85, 0.4), (84.62, 0.4), (85.8, 0.4), (86.35, 0.6), (85.92, 0.7), (85.72, 1.0)])
    # Convert standard deviations into standard errors using sqrt of number of
    # experiment replications performed
    rt_num_reps = np.sqrt(5)
    plt.figure(figsize=(6, 4))
    plt.plot(depth[start_idx:], kd_acc[start_idx:, 0], 'c-', label='KD Student')
    plt.fill_between(depth[start_idx:], kd_acc[start_idx:, 0] - kd_acc[start_idx:, 1]/rt_num_reps,
                     kd_acc[start_idx:, 0] + kd_acc[start_idx:, 1]/rt_num_reps, color='c', alpha=0.1)
    plt.plot(depth[start_idx:], kd_crossfit_acc[start_idx:, 0], 'g-', label='Cross-fit KD Student')
    plt.fill_between(depth[start_idx:], kd_crossfit_acc[start_idx:, 0] - kd_crossfit_acc[start_idx:, 1]/rt_num_reps,
                     kd_crossfit_acc[start_idx:, 0] + kd_crossfit_acc[start_idx:, 1]/rt_num_reps, color='g', alpha=0.1)
    plt.plot(depth[start_idx:], kd_crossfit_gamma_acc[start_idx:, 0], 'r-', label='Enhanced KD Student')
    plt.fill_between(depth[start_idx:], kd_crossfit_gamma_acc[start_idx:, 0] - kd_crossfit_gamma_acc[start_idx:, 1]/rt_num_reps,
                     kd_crossfit_gamma_acc[start_idx:, 0] + kd_crossfit_gamma_acc[start_idx:, 1]/rt_num_reps, color='r', alpha=0.1)
    plt.plot(depth[start_idx:], teacher_acc[start_idx:, 0], '--', color='k', label='Teacher')
    plt.fill_between(depth[start_idx:], teacher_acc[start_idx:, 0] - teacher_acc[start_idx:, 1]/rt_num_reps,
                     teacher_acc[start_idx:, 0] + teacher_acc[start_idx:, 1]/rt_num_reps, color='k', alpha=0.1)
    plt.plot(depth[start_idx:], teacher_crossfit_acc[start_idx:, 0], '--', color='m', label='Cross-fit Teacher')
    plt.fill_between(depth[start_idx:], teacher_crossfit_acc[start_idx:, 0] - teacher_crossfit_acc[start_idx:, 1]/rt_num_reps,
                     teacher_crossfit_acc[start_idx:, 0] + teacher_crossfit_acc[start_idx:, 1]/rt_num_reps, color='m', alpha=0.1)
    plt.xticks(depth[start_idx:])
    plt.yticks([82, 84, 86, 88])
    plt.ylabel('Test accuracy', fontsize=14)
    plt.xlabel("Teacher's network depth", fontsize=14)
    # plt.legend(loc='upper left')
    plt.savefig('talk_figs/cifar10_crossfit_acc.pdf', bbox_inches='tight')
    plt.close()

    kd_loss = np.array([(0.6826, 0.003), (0.5171, 0.06), (0.5519, 0.013), (0.609, 0.026), (0.6127, 0.007), (0.6322, 0.029)])
    kd_crossfit_loss = np.array([(0.6771, 0.009), (0.5088, 0.009), (0.4943, 0.007), (0.5082, 0.005), (0.5109, 0.02), (0.5285, 0.018)])
    kd_crossfit_gamma_loss = np.array([(0.6392, 0.005), (0.4982, 0.007), (0.4849, 0.017), (0.5038, 0.003), (0.5056, 0.003), (0.5200, 0.003)])
    teacher_loss = np.array([(0.8905, 0.028), (0.5038, 0.01), (0.4513, 0.017), (0.4786, 0.020), (0.4707, 0.027), (0.5147, 0.030)])
    teacher_crossfit_loss = np.array([(0.9353, 0.024), (0.5267, 0.013), (0.4969, 0.021), (0.5031, 0.017), (0.5308, 0.024), (0.547, 0.035)])
    plt.figure(figsize=(6, 4))
    # figlegend = plt.figure(figsize=(3, 4))
    plt.plot(depth[start_idx:], kd_loss[start_idx:, 0], 'c-', label='KD Student')
    plt.fill_between(depth[start_idx:], kd_loss[start_idx:, 0] - kd_loss[start_idx:, 1]/rt_num_reps,
                     kd_loss[start_idx:, 0] + kd_loss[start_idx:, 1]/rt_num_reps, color='c', alpha=0.1)
    plt.plot(depth[start_idx:], kd_crossfit_loss[start_idx:, 0], 'g-', label='Cross-fit KD Student')
    plt.fill_between(depth[start_idx:], kd_crossfit_loss[start_idx:, 0] - kd_crossfit_loss[start_idx:, 1]/rt_num_reps,
                     kd_crossfit_loss[start_idx:, 0] + kd_crossfit_loss[start_idx:, 1]/rt_num_reps, color='g', alpha=0.1)
    plt.plot(depth[start_idx:], kd_crossfit_gamma_loss[start_idx:, 0], 'r-', label=r'Enhanced KD Student')
    plt.fill_between(depth[start_idx:], kd_crossfit_gamma_loss[start_idx:, 0] - kd_crossfit_gamma_loss[start_idx:, 1]/rt_num_reps,
                     kd_crossfit_gamma_loss[start_idx:, 0] + kd_crossfit_gamma_loss[start_idx:, 1]/rt_num_reps, color='r', alpha=0.1)
    plt.plot(depth[start_idx:], teacher_loss[start_idx:, 0], '--', color='k', label='Teacher')
    plt.fill_between(depth[start_idx:], teacher_loss[start_idx:, 0] - teacher_loss[start_idx:, 1]/rt_num_reps,
                     teacher_loss[start_idx:, 0] + teacher_loss[start_idx:, 1]/rt_num_reps, color='k', alpha=0.1)
    plt.plot(depth[start_idx:], teacher_crossfit_loss[start_idx:, 0], '--', color='m', label='Cross-fit Teacher')
    plt.fill_between(depth[start_idx:], teacher_crossfit_loss[start_idx:, 0] - teacher_crossfit_loss[start_idx:, 1]/rt_num_reps,
                     teacher_crossfit_loss[start_idx:, 0] + teacher_crossfit_loss[start_idx:, 1]/rt_num_reps, color='m', alpha=0.1)
    plt.xticks(depth[start_idx:])
    plt.ylabel('Test loss', fontsize=14)
    plt.xlabel("Teacher's network depth", fontsize=14)
    plt.legend()
    # figlegend.legend()
    # figlegend.savefig('results/cifar10_crossfit_legend.pdf', bbox_inches='tight')
    plt.savefig('results/cifar10_crossfit_loss.pdf', bbox_inches='tight')
    plt.close()


def alpha_ablation():
    alpha_vals = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    kd_crossfit_gamma_acc_raw = np.array([[0.834736, 0.83754, 0.833734, 0.833734, 0.842128, 0.846755, 0.815705, 0.766627],
                                          [0.835737, 0.839744, 0.837139, 0.839543, 0.839335, 0.842548, 0.819912, 0.76262],
                                          [0.829327, 0.833734, 0.829327, 0.839744, 0.837724, 0.848958, 0.814303, 0.769431]])
    # Compute mean and standard error
    kd_mean = np.mean(kd_crossfit_gamma_acc_raw, axis=0)
    kd_std = np.std(kd_crossfit_gamma_acc_raw, axis=0)/np.sqrt(kd_crossfit_gamma_acc_raw.shape[0])
    plt.figure(figsize=(6, 4))
    plt.plot(alpha_vals, kd_mean, 'r-', label=r'Enhanced KD Student')
    plt.fill_between(alpha_vals, kd_mean - kd_std, kd_mean + kd_std, color='r', alpha=0.1)
    plt.xscale('log')
    plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], labels=[r'$0$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.yticks([0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84])
    plt.grid(axis='y')
    plt.ylabel('Test accuracy', fontsize=14)
    plt.xlabel("alpha", fontsize=14)
    plt.legend()
    # figlegend.legend()
    # figlegend.savefig('results/cifar10_crossfit_legend.pdf', bbox_inches='tight')
    plt.savefig('results/cifar10_crossfit_alpha_acc.pdf', bbox_inches='tight')
    plt.close()

    kd_crossfit_gamma_loss_raw = np.array([[0.506485, 0.513035, 0.510958, 0.510019, 0.493605, 0.493672, 0.610002, 0.875068],
                                           [0.507525, 0.501994, 0.519297, 0.503032, 0.484331, 0.49562, 0.604507, 0.847722],
                                           [0.517136, 0.4999524, 0.512288, 0.504677, 0.513491, 0.465533, 0.639482, 0.824015]])
    # Compute mean and standard error
    kd_mean = np.mean(kd_crossfit_gamma_loss_raw, axis=0)
    kd_std = np.std(kd_crossfit_gamma_loss_raw, axis=0)/np.sqrt(kd_crossfit_gamma_loss_raw.shape[0])
    plt.figure(figsize=(6, 4))
    plt.plot(alpha_vals, kd_mean, 'r-', label=r'Enhanced KD Student')
    plt.fill_between(alpha_vals, kd_mean - kd_std, kd_mean + kd_std, color='r', alpha=0.1)
    plt.xscale('log')
    plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], labels=[r'$0$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    # plt.yticks([0.50, 0.525, 0.55, 0.65, 0.75, 0.85])
    plt.grid(axis='y')
    plt.ylabel('Test loss', fontsize=14)
    plt.xlabel("alpha", fontsize=14)
    plt.legend()
    # figlegend.legend()
    # figlegend.savefig('results/cifar10_crossfit_legend.pdf', bbox_inches='tight')
    plt.savefig('results/cifar10_crossfit_alpha_loss.pdf', bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()
    alpha_ablation()
