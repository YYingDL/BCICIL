import numpy as np
from scipy.stats import sem
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:        3D nd_arrays  (num_runs, num_tasks, num_tasks)
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)  # t coefficient used to compute 95% CIs: mean +- t*SD, alpha/2 = 0.975

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_tasks)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run)) if n_run > 1 else (np.mean(avg_acc_per_run),)

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)  # (num_run, num_tasks)
    final_forgets = best_acc - end_acc

    avg_fgt = np.mean(final_forgets[:, :-1], axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt)) if n_run > 1 else (np.mean(avg_fgt),)

    # compute Avg ACC on old tasks of each task
    acc_per_run = (np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1))
    avg_acc = (np.mean(acc_per_run, axis=0), t_coef * sem(acc_per_run, axis=0)) if n_run > 1 else (np.mean(acc_per_run, axis=0),)

    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run)) if n_run > 1 else (np.mean(bwtp_per_run),)

    # compute Avg Acc_cur (diagonal elements)
    diagonals = []
    for i in range(n_run):
        matrix = end_task_acc_arr[i]
        diagonal = np.diag(matrix)
        diagonals.append(diagonal)
    diagonal_means = [np.mean(diag) for diag in diagonals]
    avg_cur_acc = (np.mean(diagonal_means), t_coef * sem(diagonal_means)) if n_run > 1 else (np.mean(diagonal_means), )

    return avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc, avg_bwtp


def compute_performance_offline(acc_multiple_run):
    n_run = acc_multiple_run.shape[0]

    t_coef = stats.t.ppf((1 + 0.95) / 2, n_run - 1)

    # compute average test accuracy and CI
    acc = (np.mean(acc_multiple_run), t_coef * sem(acc_multiple_run)) if n_run > 1 else (np.mean(acc_multiple_run),)
    return acc


def single_run_avg_end_fgt(acc_array):
    best_acc = np.max(acc_array, axis=1)
    end_acc = acc_array[-1]
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets)
    return avg_fgt


def plot_confusion_matrix(y_true, y_pred, classes, path, title=None):
    """绘制混淆矩阵热力图

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别列表
        path: 保存路径
        title: 图表标题（可选）
    """
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix / np.sum(cf_matrix, axis=1, keepdims=True)
    cf_matrix = np.around(cf_matrix, decimals=2)
    np.save(path, cf_matrix)
    import pandas as pd
    import seaborn as sn
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])

    # Dynamically adjust figure size based on number of classes
    n_classes = len(classes)

    # 根据类别数量调整图像大小和字体
    if n_classes <= 10:
        figsize = (max(6, n_classes * 0.8), max(6, n_classes * 0.7))
        annot_font_size = 10
        label_font_size = 12
        x_rotation = 45
    elif n_classes <= 20:
        figsize = (max(8, n_classes * 0.6), max(8, n_classes * 0.6))
        annot_font_size = 8
        label_font_size = 10
        x_rotation = 45
    elif n_classes <= 40:
        figsize = (max(10, n_classes * 0.5), max(10, n_classes * 0.5))
        annot_font_size = 6
        label_font_size = 8
        x_rotation = 90
    else:  # 40+ 类别
        figsize = (max(12, n_classes * 0.45), max(12, n_classes * 0.45))
        annot_font_size = 5
        label_font_size = 7
        x_rotation = 90

    plt.figure(figsize=figsize)

    s = sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="YlOrBr", center=0.3, square=True,
                   annot_kws={'size': annot_font_size})
    s.set(xlabel='Prediction', ylabel='Ground truth')
    s.tick_params(axis='both', labelsize=label_font_size)

    # Add title if provided
    if title:
        plt.title(title, fontsize=max(10, 14 - n_classes * 0.1))

    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=x_rotation, ha='right' if x_rotation == 45 else 'center')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_confusion_matrix_direct(cf_matrix, path, classes, title=None):
    """直接绘制已计算好的混淆矩阵

    Args:
        cf_matrix: 已归一化的混淆矩阵
        path: 保存路径
        classes: 类别列表
        title: 图表标题（可选）
    """
    np.save(path, cf_matrix)
    import pandas as pd
    import seaborn as sn
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])

    # Dynamically adjust figure size based on number of classes
    n_classes = len(classes)

    # 根据类别数量调整图像大小和字体
    if n_classes <= 10:
        figsize = (max(6, n_classes * 0.8), max(6, n_classes * 0.7))
        annot_font_size = 10
        label_font_size = 12
        x_rotation = 45
    elif n_classes <= 20:
        figsize = (max(8, n_classes * 0.6), max(8, n_classes * 0.6))
        annot_font_size = 8
        label_font_size = 10
        x_rotation = 45
    elif n_classes <= 40:
        figsize = (max(10, n_classes * 0.5), max(10, n_classes * 0.5))
        annot_font_size = 6
        label_font_size = 8
        x_rotation = 90
    else:  # 40+ 类别
        figsize = (max(12, n_classes * 0.45), max(12, n_classes * 0.45))
        annot_font_size = 5
        label_font_size = 7
        x_rotation = 90

    plt.figure(figsize=figsize)

    s = sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="YlOrBr", center=0.3, square=True,
                   annot_kws={'size': annot_font_size})
    s.set(xlabel='Prediction', ylabel='Ground truth')
    s.tick_params(axis='both', labelsize=label_font_size)

    # Add title if provided
    if title:
        plt.title(title, fontsize=max(10, 14 - n_classes * 0.1))

    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=x_rotation, ha='right' if x_rotation == 45 else 'center')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.show()
