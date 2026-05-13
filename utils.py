import pdb
import sys
from sklearn import metrics
import numpy as np
import torch

seed_dict = {'abu-beach-1': 2369, 'Pavia_100': 4521, 'GrandIsle': 2,
             'MUUFLGulfport': 8, 'UHAD-U-I': 4857, 'UHAD-U-II': 782, }


class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def get_params(net):
    """Returns parameters that we want to optimize over.
    """
    params = []
    params += [x for x in net.parameters()]

    return params


def img2mask(img):
    img = img[0].sum(0)
    img = img - img.min()
    img = img / img.max()
    img = img.detach().cpu().numpy()

    return img


def Residual(contr_data, org_data):
    row, col, band = org_data.shape
    residual = np.square(org_data - contr_data)
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            R = np.mean(residual[i, j, :])
            result[i, j] = R

    return result


def ROC_AUC(target2d, groundtruth):
    """
    :param target2d: the 2D anomaly component
    :param groundtruth: the groundtruth
    :return: auc: the AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    target2d = target2d.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    print('The AUC Value: ', auc)

    return auc


def Mahalanobis(data):
    row, col, band = data.shape
    data = data.reshape(row * col, band)
    mean_vector = np.mean(data, axis=0)
    mean_matrix = np.tile(mean_vector, (row * col, 1))
    re_matrix = data - mean_matrix
    matrix = np.dot(re_matrix.T, re_matrix) / (row * col - 1)
    variance_covariance = np.linalg.pinv(matrix)

    distances = np.zeros([row * col, 1])
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = np.dot(re_array, variance_covariance)
        distances[i] = np.dot(re_var, np.transpose(re_array))
    distances = distances.reshape(row, col)

    return distances


def false_alarm_rate(target, predicted):
    target = ((target - target.min()) /
              (target.max() - target.min()))
    predicted = ((predicted - predicted.min()) /
                 (predicted.max() - predicted.min()))
    anomaly_map = target
    normal_map = 1 - target
    num = 30000
    taus = np.linspace(0, predicted.max(), num=num)
    PF = np.zeros([num, 1])
    PD = np.zeros([num, 1])
    for index in range(num):
        tau = taus[index]
        anomaly_map_1 = np.double(predicted >= tau)
        PF[index] = np.sum(anomaly_map_1 * normal_map) / np.sum(normal_map)
        PD[index] = np.sum(anomaly_map_1 * anomaly_map) / np.sum(anomaly_map)
    PD_PF_auc = np.sum((PF[0:num - 1, :] - PF[1:num, :]) * (PD[1:num] + PD[0:num - 1]) / 2)
    # area0 = np.trapz(PD.squeeze(),PF.squeeze())
    PF_tau_auc = np.trapz(PF.squeeze(), taus.squeeze())
    return PD_PF_auc, PF_tau_auc
