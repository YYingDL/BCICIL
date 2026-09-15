# -*- coding: UTF-8 -*-
import numpy as np
from utils.data import extract_samples_according_to_labels, extract_samples_according_to_labels_with_sub, data_alignment
from utils.utils import load_pickle
from utils.setup_elements import n_classes, n_tasks, data_path, preset_orders, n_tasks, n_classes_per_task, input_size_match,n_classes_init_task
import torch
import random


class IncrementalTaskStream(object):
    def __init__(self, data, scenario, cls_order, split):

        self.tasks = []
        self.data = data
        self.scenario = scenario
        self.path = data_path[data]
        self.split = split

        if self.split == 'all':  # Decide which task stream to use, Exp or Val or All
            start, end = 0, n_tasks[data]
        elif self.split == 'val':
            start, end = 0, n_tasks[data]
        elif self.split == 'exp':
            start = 0 if n_tasks[self.data] <= 5 else n_tasks[data]
            end = n_tasks[data]
        else:
            raise ValueError("Incorrect task stream split")

        # Setup elements for this task stream
        self.n_tasks = end - start
        self.n_class_per_task = n_classes_per_task[data]
        self.n_classes = n_classes[data]
        self.init_order_list = cls_order[:n_classes_init_task[data]]
        self.order_list = [i for i in range(4)]
        print("Create {} stream : {} tasks,  classes order {} ".format(self.split, self.n_tasks, self.init_order_list))
        print("Input shape (L, D): {}".format(tuple(input_size_match[data])))

    def load_data(self, load_subject=False):
        """
        Load data from .pkl files into np arrays.
        For methods using subject labels, set load_subject = True
        """

        x_train = load_pickle(self.path + 'x_train.pkl')
        y_train = load_pickle(self.path + 'state_train.pkl')
        x_test = load_pickle(self.path + 'x_test.pkl')
        y_test = load_pickle(self.path + 'state_test.pkl')

        if load_subject:
            sub_train = load_pickle(self.path + 'subject_label_train.pkl')
            sub_train = sub_train.squeeze()
        else:
            sub_train = None

        if n_tasks[self.data] < 5 and self.split == 'val':
            x_test = load_pickle(self.path + 'x_val.pkl')
            y_test = load_pickle(self.path + 'state_val.pkl')

        return x_train, y_train.squeeze(), x_test, y_test.squeeze(), sub_train

    def setup(self, cut=0.9, load_subject=False):
        """
        Arrange the data into tasks, according to the class order.
        Each task has a train set, val set (for earlystop) and test set
        If load_subject=True, train set contains subject labels.
        """

        x_train, y_train, x_test, y_test, sub_train = self.load_data(load_subject)


        class_idx = 0
        train_size, test_size, val_size = 0, 0, 0
        for t in range(self.n_tasks):
            classes_in_task_t = []  # class index that contains in task t
            if len(classes_in_task_t) == 0:
                classes_in_task_t.append(self.init_order_list)
            else:
                classes_in_task_t.append(len(classes_in_task_t) + np.arange(self.n_class_per_task))


            if load_subject:
                x_train_t, y_train_t, sub_train_t = extract_samples_according_to_labels_with_sub(x_train, y_train, sub_train, classes_in_task_t)
            else:
                x_train_t, y_train_t = extract_samples_according_to_labels(x_train, y_train, classes_in_task_t)

            x_test_t, y_test_t = extract_samples_according_to_labels(x_test, y_test, classes_in_task_t)

            if self.scenario == 'class':
                pass
                # map the class labels to the ordered ones
                # y_train_t = np.array([self.order_list.index(i) for i in y_train_t])
                # y_test_t = np.array([self.order_list.index(i) for i in y_test_t])

            elif self.scenario == 'domain':
                # TODO: extend it to DIL in the future
                pass

            if load_subject:
                train_data = (x_train_t, y_train_t, sub_train_t)
                train_data, val_data = make_valid_from_train_with_sub(train_data, cut=cut)
            else:
                train_data = (x_train_t, y_train_t)
                train_data, val_data = make_valid_from_train(train_data, cut=cut)
            test_data = (x_test_t, y_test_t)

            train_size += train_data[0].shape[0]
            val_size += val_data[0].shape[0]
            test_size += test_data[0].shape[0]

            self.tasks.append((train_data, val_data, test_data))

        print("Training set size: {}; Val set size: {}; Test set size: {}".format(train_size, val_size, test_size))

    def setup_offline(self, cut=0.9, load_subjects=0):
        x_train, y_train, x_test, y_test, sub_train = self.load_data(False)
        train_idxs = list(range(x_train.shape[0]))
        del train_idxs[load_subjects]
        x_train, y_train = x_train[train_idxs].reshape(-1, x_train.shape[-2], x_train.shape[-1]), y_train[
            train_idxs].reshape(-1, 1)
        x_test, y_test = x_test[load_subjects].reshape(-1, x_test.shape[-2], x_test.shape[-1]), y_test[
            load_subjects].reshape(-1, 1)
        if self.data == 'mi':
            x_train = data_alignment(x_train, num_subjects=len(train_idxs))
            x_test = data_alignment(x_test, num_subjects=1)
        if self.data == 'seed3':
            if self.init_order_list == [0, 2]:
                mapping = np.array([0, 2, 1])
            elif self.init_order_list == [1, 2]:
                mapping = np.array([2, 0, 1])
            y_train, y_test = mapping[y_train], mapping[y_test]
            self.init_order_list = [0, 1]

        x_train, y_train = extract_samples_according_to_labels(x_train, y_train, self.order_list)
        x_test, y_test = extract_samples_according_to_labels(x_test, y_test, self.order_list)

        # map the class labels to the ordered ones
        y_train = np.array([self.order_list.index(i) for i in y_train])
        y_test = np.array([self.order_list.index(i) for i in y_test])

        train_data = (x_train, y_train)
        test_data = (x_test, y_test)
        train_data, val_data = train_data, test_data

        train_size = train_data[0].shape[0]
        val_size = val_data[0].shape[0]
        test_size = test_data[0].shape[0]
        print("Training set size: {}; Val set size: {}; Test set size: {}".format(train_size, val_size, test_size))

        return train_data, val_data, test_data

    def setup_subject(self, cut=0.9, load_subjects=0):
        """
        Arrange the data into tasks, according to the class order.
        Each task has a train set, val set (for earlystop) and test set
        If load_subject=True, train set contains subject labels.
        """

        x_train, y_train, x_test, y_test, sub_train = self.load_data(False)
        train_idxs = list(range(x_train.shape[0]))
        del train_idxs[load_subjects]
        x_train, y_train = x_train[train_idxs].reshape(-1, x_train.shape[-2],x_train.shape[-1]), y_train[train_idxs].reshape(-1, 1)
        x_test, y_test = x_test[load_subjects].reshape(-1, x_test.shape[-2],x_test.shape[-1]), y_test[load_subjects].reshape(-1, 1)
        if self.data == 'mi':
            x_train = data_alignment(x_train, num_subjects=len(train_idxs))
            x_test = data_alignment(x_test, num_subjects=1)
        if self.data == 'seed3':
            if self.init_order_list == [0, 2]:
                mapping = np.array([0, 2, 1])
            elif self.init_order_list == [1, 2]:
                mapping = np.array([2, 0, 1])
            elif self.init_order_list == [0, 1]:
                mapping = np.array([0, 1, 2])
            y_train, y_test = mapping[y_train],mapping[y_test]
            self.init_order_list = [0, 1]
        # if self.data =='benchmark':
        #     x_train, x_test = filterbank(x_train), filterbank(x_test)
        class_idx = 0
        train_size, test_size, val_size = 0, 0, 0
        for t in range(self.n_tasks):
            classes_in_task_t = []  # class index that contains in task t

            if t == 0:
                classes_in_task_t = self.init_order_list
            else:
                classes_in_task_t = [i+len(self.init_order_list)+class_idx*self.n_class_per_task for i in range(self.n_class_per_task)]
                class_idx += 1

            x_train_t, y_train_t = extract_samples_according_to_labels(x_train, y_train, classes_in_task_t)

            x_test_t, y_test_t = extract_samples_according_to_labels(x_test, y_test, classes_in_task_t)

            if self.scenario == 'class':
                y_train_t = np.array([i for i in y_train_t]).reshape(-1)
                y_test_t = np.array([i for i in y_test_t]).reshape(-1)

            elif self.scenario == 'domain':
                # TODO: extend it to DIL in the future
                pass


            train_data = (x_train_t, y_train_t)
            test_data = (x_test_t, y_test_t)
            train_data, val_data = train_data, test_data
            # train_data, val_data = make_valid_from_train(train_data, cut=cut)

            train_size += train_data[0].shape[0]
            val_size += val_data[0].shape[0]
            test_size += test_data[0].shape[0]

            self.tasks.append((train_data, val_data, test_data))

        print("Training set size: {}; Val set size: {}; Test set size: {}".format(train_size, val_size, test_size))


def make_valid_from_train(dataset, cut=0.95):
    x_t, y_t = dataset
    x_tr, y_tr, x_val, y_val = [], [], [], []
    for cls in set(y_t.tolist()):
        x_cls, y_cls = extract_samples_according_to_labels(x_t, y_t, [cls])
        perm = torch.randperm(len(x_cls))
        x_cls, y_cls = x_cls[perm], y_cls[perm]
        split = int(len(x_cls) * cut)
        x_tr_cls, y_tr_cls = x_cls[:split], y_cls[:split]
        x_val_cls, y_val_cls = x_cls[split:], y_cls[split:]

        x_tr.append(x_tr_cls)
        y_tr.append(y_tr_cls)
        x_val.append(x_val_cls)
        y_val.append(y_val_cls)

    x_tr = np.concatenate(x_tr)
    x_val = np.concatenate(x_val)
    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)
    return (x_tr, y_tr), (x_val, y_val)


def make_valid_from_train_with_sub(dataset, cut=0.9):
    x_t, y_t, sub_t = dataset
    x_tr, y_tr, x_val, y_val, sub_tr, sub_val = [], [], [], [], [], []
    for cls in set(y_t.tolist()):
        x_cls, y_cls, sub_cls = extract_samples_according_to_labels_with_sub(x_t, y_t, sub_t, [cls])
        perm = torch.randperm(len(x_cls))
        x_cls, y_cls, sub_cls = x_cls[perm], y_cls[perm], sub_cls[perm]
        split = int(len(x_cls) * cut)
        x_tr_cls, y_tr_cls, sub_tr_cls = x_cls[:split], y_cls[:split], sub_cls[:split]
        x_val_cls, y_val_cls, sub_val_cls = x_cls[split:], y_cls[split:], sub_cls[split:]

        x_tr.append(x_tr_cls)
        y_tr.append(y_tr_cls)
        sub_tr.append(sub_tr_cls)
        x_val.append(x_val_cls)
        y_val.append(y_val_cls)

    x_tr = np.concatenate(x_tr)
    x_val = np.concatenate(x_val)
    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)
    sub_tr = np.concatenate(sub_tr)

    return (x_tr, y_tr, sub_tr), (x_val, y_val)


def get_cls_order(data, fix_order=True):
    if fix_order:
        return preset_orders[data]
    else:
        all_classes = np.arange(n_classes[data])
        # np.random.shuffle(all_classes)
        cls_order = list(all_classes)
        return cls_order


def filterbank(X, num_subbands=3):
    """
    Suggested filterbank function for benchmark dataset
    """

    from scipy import signal
    srate = 250
    X = np.transpose(X,(0, 2, 1))
    filterbank_X = np.zeros((X.shape[0], num_subbands, X.shape[-2], X.shape[-1]))
    for i in range(X.shape[0]):
        x_tmp = X[i]
        for k in range(1, num_subbands + 1, 1):
            Wp = [(8 * k) / (srate / 2), 90 / (srate / 2)]
            Ws = [(8 * k - 2) / (srate / 2), 100 / (srate / 2)]
            gstop = 40
            while gstop >= 20:
                try:
                    N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                    bpB, bpA = signal.cheby1(N, 0.5, Wn, btype='bandpass')
                    filterbank_X[i, k - 1, :, :] = signal.filtfilt(bpB, bpA, x_tmp, axis=1, padtype='odd',
                                                                   padlen=3 * (max(len(bpB), len(bpA)) - 1))
                    break
                except:
                    gstop -= 1
    return np.transpose(filterbank_X,(0, 1, 3, 2))