import numpy as np
from numpy import dstack
from pandas import read_csv, read_table
import pickle
from scipy.io import loadmat
import torch
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JFPAA_net parameters')
    parser.add_argument('--dataset', type=str, default='benchmark',
                        help='the dataset used for SSVEP, "benchmark" or "beta"')
    parser.add_argument('--batch_size', type=int, default=128, help='size for one batch, integer')
    parser.add_argument('--data_length', type=int, default=int(250 * 1.0), help='size for data_length, integer')
    parser.add_argument('--num_class', type=int, default=40, help='number for target, integer')
    parser.add_argument('--epochs', type=int, default=100, help='training epoch, integer')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--model_name', default='ADFCNN', type=str,
                        help='the current model name: [CCA, FBCCA, MSI, TRCA, EEGNet, ShallowConvNet, GuneyDNN, CCNN, FBSSVEPFormer, DBACFNet,'
                             'FBDBACFNet, ADFCNN]')
    parser.add_argument('--output_log_dir', default='./train_log', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--device', type=str, default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                        help='learning rate')
    args = parser.parse_args()
    args.dataset = 'benchmark'
    path1 ='/home/user_yy/Dataset/Benchmark Dataset/'
    path2 = '.mat'
    channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]
    name = ['S' + str(i) for i in range(1, 36)]
    t_delay = int((0.5 + 0.12) * 250)
    block = 6
    subband = 3
    frequencySet = np.array(
        [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0,
         12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2,
         15.4, 15.6, 15.8])
    phaseSet = np.array(
        [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
         1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]) * np.pi


    index_class = range(0, 40)
    for i in range(len(name)):
        path = path1 + name[i] + path2
        mat = loadmat(path)
        if args.dataset == 'beta':
            data1c = mat['data']['EEG'][0][0]
            data1c = data1c[channels, t_delay:t_delay + args.data_length, :, :args.num_class]
            data1c = data1c.transpose(3, 2, 0, 1)
        else:
            data1c = mat['data']
            data1c = data1c[channels, t_delay:t_delay + args.data_length, :args.num_class, :]
            data1c = data1c.transpose(2, 3, 0, 1)  # label*block*C*T
        data_tmp = np.zeros((block * args.num_class, data1c.shape[2], data1c.shape[3]))
        label_tmp = np.zeros(block * args.num_class)
        for j in range(args.num_class):
            data_tmp[block * j:block * j + block] = data1c[j]
            label_tmp[block * j:block * j + block] = np.ones(block) * j

        if (i == 0):
            train_datac = data_tmp
            train_labelc = label_tmp
        else:
            train_datac = np.append(train_datac, data_tmp, axis=0)
            train_labelc = np.append(train_labelc, label_tmp)

    data_session = train_datac.reshape((35, 240, 9, 250)).transpose(0, 1, 3, 2)
    label_session = train_labelc.reshape((35, 240))
    path = './saved/benchmark/'
    if not os.path.exists(path):
        os.makedirs(path)
    import pickle

    with open(path + '/x_train.pkl', 'wb') as f:
        pickle.dump(data_session, f)  # trainX tasks around 500 MB
    with open(path + '/state_train.pkl', 'wb') as f:
        pickle.dump(label_session, f)

    with open(path + '/x_test.pkl', 'wb') as f:
        pickle.dump(data_session, f)
    with open(path + '/state_test.pkl', 'wb') as f:
        pickle.dump(label_session, f)

