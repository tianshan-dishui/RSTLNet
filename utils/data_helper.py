from math import sqrt
import os
import random
import sys

sys.path.append(os.getcwd())
import torch
import numpy as np
from torch.utils.data import DataLoader
from RSTLNet.utils.data_process import np_to_tensor_dataset, \
    split_dataset_with_imputer


def mask_generation(dataset='abilene', ratio=0.35, counts=3):
    for i in range(counts):
        if dataset == 'abilene':
            shape = (48096, 144)
        elif dataset == 'geant':
            shape = (10772, 529)
        else:
            return

        num_ones = int(ratio * np.prod(shape))
        indices = np.random.permutation(np.prod(shape))
        mask = np.zeros(shape, dtype=int)
        mask.ravel()[indices[:num_ones]] = 1
        np.save(dataset + str(ratio) + '_' + str(i + 1) + '.npy', mask)

def get_data_path(dataset='abilene'):
    dataset_path = os.path.dirname(os.path.dirname(__file__))
    data_path = None
    if dataset == 'geant':
        data_path = 'geant.npy'
    elif dataset == 'abilene':
        data_path = 'abilene.npy'
    data_path = os.path.join(dataset_path, 'dataset', data_path)
    if data_path is None:
        print("Can not get the {dataset} dataset.".format(dataset))
        print("So get the Abilene dataset.")
        return get_data_path()
    print("Get the {d} dataset at {p}.".format(d=dataset, p=data_path))
    return data_path

def get_dataset_nodes(dataset='geant'):
    nodes = 12
    if dataset == 'geant':
        nodes = 23
    elif dataset == 'abilene':
        nodes = 12
    return nodes, nodes ** 2


def get_adj_matrix(dataset='abilene'):
    project_path = os.path.dirname(os.path.dirname(__file__))
    adj_path = dataset + '_adj.npy'
    adj_path = os.path.join(project_path, 'topo', adj_path)
    return adj_path


def change_topo_matrix(matrix):
    row_sums = matrix.sum(axis=1)
    normalized_matrix = np.divide(matrix, row_sums[:, np.newaxis], out=np.zeros_like(matrix),
                                  where=row_sums[:, np.newaxis] != 0)
    return normalized_matrix


# get dataloader
def get_dataloaders(data_path=None, train_rate=0.6, test_rate=0.2, seq_len=24, pre_len=1, sw_step=1, missing_ratio=0.6,
                    missing_index=1, batch_size=64, num_workers=8, imputer='mean', random=None, test=False):
    dataloaders = {}

    train_x, train_y, val_x, val_y, test_x, test_y, max_value, er = split_dataset_with_imputer(data_path,
                                                                                               train_rate=train_rate,
                                                                                               test_rate=test_rate,
                                                                                               seq_len=seq_len,
                                                                                               predict_len=pre_len,
                                                                                               sw_step=sw_step,
                                                                                               missing_ratio=missing_ratio,
                                                                                               missing_index=missing_index,
                                                                                               test=test,
                                                                                               imputer=imputer
                                                                                               )
    dataloaders['er'] = er
    dataloaders['time'] = er

    dataset = np_to_tensor_dataset(train_x, train_y)
    dataloaders['train'] = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=num_workers)
    dataset = np_to_tensor_dataset(val_x, val_y)
    dataloaders['val'] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                    num_workers=num_workers)
    dataset = np_to_tensor_dataset(test_x, test_y)
    dataloaders['test'] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                     num_workers=num_workers)
    return dataloaders


if __name__ == '__main__':
    pass
