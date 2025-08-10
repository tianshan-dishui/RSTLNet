import math
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def ERROR_RATIO(pred, target, mask):
    """
    :param pred: predictions
    :param target: ground truth
    :param mask: mask matrix
    :return: error ratio
    """
    er = torch.sqrt(torch.sum(((pred - target) * mask) ** 2.0)) / torch.sqrt(torch.sum((target * mask) ** 2.0))
    return er


def R2(pred, y, mask=None):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """

    # temp_r2 = 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(y,dim=0)) ** 2)
    # temp_r2 = 0.
    # for i in range(y.shape[1]):
    #     temp_r2 += r2_score(y[:,i].cpu(),pred[:,i].cpu())
    # temp_r2 /= y.shape[1]
    # r2 = r2_score(y.flatten().cpu(),pred.flatten().cpu())
    # print(temp_r2,r2)
    # return r2
    if mask is not None:
        mean = torch.sum(y * mask) / torch.sum(mask)
        return 1 - torch.sum(((y - pred) * mask) ** 2) / torch.sum(((y - mean) * mask) ** 2)
    else:

        return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(y)) ** 2)


def MSE(pred, y, mask=None):
    if mask is not None:
        mask = mask.bool()
        true_missing = y[mask]
        filled_missing = pred[mask]
        errors = true_missing - filled_missing
        mse = torch.mean(errors ** 2)
        return mse
    else:
        return torch.mean((pred - y) ** 2)


def RMSE(pred, y):
    this_type_str = type(pred)
    if this_type_str is np.ndarray:
        pred = torch.tensor(pred)
        y = torch.tensor(y)
    return torch.sqrt(torch.mean((pred - y) ** 2))


def MAE(pred, y, mask=None):
    if mask is not None:
        mask = mask.bool()
        true_missing = y[mask]
        filled_missing = pred[mask]
        errors = true_missing - filled_missing
        mae = torch.mean(torch.abs(errors))
        return mae
    else:
        return torch.mean(torch.abs(pred - y))


def MAPE(pred, y, epsilon=1e-9):
    this_type_str = type(pred)
    if this_type_str is np.ndarray:
        pred = torch.tensor(pred, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

    nonzero_indices = y != 0
    pred_filtered = pred[nonzero_indices]
    y_filtered = y[nonzero_indices]

    # y_with_epsilon = y_filtered + epsilon
    y_with_epsilon = y_filtered
    ape = torch.abs((pred_filtered - y_filtered) / y_with_epsilon)
    mape = torch.mean(ape) * 100
    return mape


