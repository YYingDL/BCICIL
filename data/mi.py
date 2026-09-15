import os
import numpy as np
import sklearn.preprocessing as preprocessing
import scipy.io as scio
import random


def data_process(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('E:/dataset/' + 'BNCI2014001' + '/X.npy')
        y = np.load('E:/dataset/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('E:/dataset/' + dataset + '/X.npy')
        y = np.load('E:/dataset/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


if __name__ == "__main__":
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process('BNCI2014001-4')

    data_session, label_session = np.array(X).reshape(9,288,22,1001).transpose(0,1,3,2), np.array(y).reshape(9,288)
    # data_session = data_session.reshape(15,3394, 62, 5)
    path = './saved/mi/'
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
