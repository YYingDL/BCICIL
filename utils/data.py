# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import fractional_matrix_power


def data_alignment(X, num_subjects):
    """数据对齐"""
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    return X

def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    x = x.transpose(0, 2, 1)
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA.transpose(0, 2, 1)


def Dataloader_from_numpy(X, Y, batch_size, shuffle=True):
    """
    - Tensors in dataloader are in cpu.
    """
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def Dataloader_from_numpy_with_idx(X, idx, Y, batch_size, shuffle=True):
    """
    - Tensors in dataloader are in cpu.
    """
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(idx).long(), torch.Tensor(Y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def Dataloader_from_numpy_with_sub(X, Y, Sub, batch_size, shuffle=True):
    """
    - Tensors in dataloader are in cpu.
    """
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).long(), torch.Tensor(Sub).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def extract_samples_according_to_labels(x, y, target_ids, return_inds=False):
    """
    Extract corresponding samples from x and y according to the labels
    :param x: data, np array
    :param y: labels, np array
    :param target_ids: list of labels
    :return:
    """
    # get the indices
    inds = list(map(lambda x: x in target_ids, y))
    x_extracted = x[inds]
    y_extracted = y[inds]

    if return_inds:
        return x_extracted, y_extracted, inds
    else:
        return x_extracted, y_extracted


def extract_samples_according_to_labels_with_sub(x, y, sub, target_ids, return_inds=False):
    """
    Extract corresponding samples with subject label from x and y according to the labels
    :param x: data, np array
    :param y: labels, np array
    :param sub: subject labels, np array
    :param target_ids: list of labels
    :return:
    """
    # get the indices
    inds = list(map(lambda x: x in target_ids, y))
    x_extracted = x[inds]
    y_extracted = y[inds]
    sub_extracted = sub[inds]

    if return_inds:
        return x_extracted, y_extracted, sub_extracted, inds
    else:
        return x_extracted, y_extracted, sub_extracted


def extract_samples_according_to_subjects(x, y, sub, target_ids, return_inds=False):
    # get the indices
    inds = list(map(lambda x: x in target_ids, sub))
    x_extracted = x[inds]
    y_extracted = y[inds]

    if return_inds:
        return x_extracted, y_extracted, inds
    else:
        return x_extracted, y_extracted


def extract_n_samples_randomly(x, y, n_sample):
    """
    Randomly extract n samples from x and y
    :param x: data, np array
    :param y: labels, np array
    :param n_sample: Number of samples to extract
    :return: extracted data & labels
    """
    sampled_idx = np.random.randint(0, len(y), size=n_sample)
    return x[sampled_idx], y[sampled_idx]

