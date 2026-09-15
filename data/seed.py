import os
import numpy as np
import sklearn.preprocessing as preprocessing
import scipy.io as scio
import random


def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type}
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    if dataset_name == 'Seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'Seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')


def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    data: [channels, samples, frequency] to [samples, frequency, channels]
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        # one_data = np.reshape(np.transpose(
        #     data[i], (1, 2, 0)), (-1, 310))
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310))
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label


def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}:
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
                   value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    # we will have a preprocessing here
    one_sub_data = one_sub_data.astype(np.float32)
    one_sub_label = one_sub_label.astype(np.int64)
    # one_sub_data = norminy(one_sub_data)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    one_sub_data = min_max_scaler.fit_transform(one_sub_data).astype(np.float32)
    return one_sub_data, one_sub_label


def sample_by_value(list, value, number):
    '''
    @Description: sample the given list randomly with given value
    @param {type}:
        list: list
        value: int {0,1,2,3}
        number: number of sampling
    @return:
        result_index: list
    '''
    result_index = []
    index_for_value = [i for (i, v) in enumerate(list) if v == value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index


def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)
    '''
    dataset_path = {'Seed4': '/home/user_yy/Dataset/seed_iv', 'Seed3': 'E:/dataset/Seed/Feature extraction',
                    'deafseed3':'/home/user_yy/Dataset/deafseed'}
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats = mats[6:] + mats[:6]
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats


if __name__ == "__main__":
    data_name= 'Seed3' #, 'Seed4'
    path, allmats = get_allmats_name(data_name)
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/' + str(i + 1) + '/' + allmats[i][j]
            one_data, one_label = get_data_label_frommat(
                mat_path, data_name, i)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()

    data_session, label_session = np.array(data[0]), np.array(label[0])
    data_session = data_session.reshape(15,3394, 62, 5)
    path = './saved/Seed/'
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
