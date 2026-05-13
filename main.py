import os
import numpy as np
import torch
import torch.optim as Optim
import scipy.io as sio
import pdb
from MrsNet import MrsNet
from sklearn.metrics import roc_auc_score, roc_curve
import shutil
from utils import get_params, img2mask, seed_dict, ROC_AUC, Mahalanobis, Residual, false_alarm_rate
import random
import time
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from Mydata import Data


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_dir = './data/'
save_dir = './results/'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False


def main(file):
    # *********************************************************************************************************
    # set random seed
    seed = seed_dict[file]
    set_seed(seed)
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    # load data
    mat = sio.loadmat(data_path)
    img_var = mat['data']
    gt = mat['map']
    row, col, band = img_var.shape
    # *********************************************************************************************************
    # 设置参数
    LR = 0.1
    end_iter = 20
    # 可调参数 #######################################################
    patch_size = 7
    spa_rate = 0.2
    spe_window = 7
    batch_size = 2 ** 10
    hidden_node = 18
    # ###############################################################
    data_set = Data(img=img_var, wout_size=patch_size, spatial_mask_rate=spa_rate,
                    spectral_window=spe_window)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=False)
    # *********************************************************************************************************
    # model setup
    net = MrsNet(in_ch=band, out_ch=band, head_ch=hidden_node).to(device)
    # loss
    mse = torch.nn.MSELoss(reduction='mean')
    L1 = torch.nn.L1Loss()
    # optim
    p = get_params(net)
    optimizer = Optim.Adam(p, lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopper = EarlyStopper(patience=5, min_delta=0.0001)
    print('Starting optimization')

    # **********************************************************************************************************
    # start train
    start = time.time()
    for iter in range(1, end_iter + 1):
        running_loss = 0.0
        net.train()
        loss = 0
        for idx, batch_data in enumerate(data_loader):
            # input -> net -> output
            code_center, window, mask_window = batch_data
            net_out = net(mask_window.to(device), code_center.to(device))
            # cal loss
            loss = mse(net_out.cpu(), window)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()
            torch.save(net, './checkpoint/{}_model.pth'.format(file))
        print('Epoch:', iter, '| train loss: %.4f' % loss.data.cpu().numpy())
        epoch_loss = running_loss / batch_size
        if early_stopper.should_stop(epoch_loss):
            print(f'Early stopping triggered at epoch {iter + 1}')
            break

    # *****************************************************************************************************
    # start test
    infer_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, drop_last=False)
    infer_res_list = []
    for idx, data in enumerate(infer_loader):
        code_center, window, mask_window = data
        # inference
        with torch.no_grad():
            net = torch.load('./checkpoint/{}_model.pth'.format(file))
            infer_out = net(window.to(device), code_center.to(device))
            infer_res_list.append(infer_out.cpu())
    infer_out = torch.cat(infer_res_list, dim=0)
    out = infer_out[:, :, int(patch_size / 2), int(patch_size / 2)]
    out = out.detach().numpy()
    re_img = np.reshape(out, (row, col, band))
    sio.savemat('./re_data/{}_redata.mat'.format(file), {'data': re_img})

    # ****************************************************************************************************
    # running time
    end = time.time()
    print("Runtime：%.4f" % (end - start))

    # 对数据进行归一化处理
    img_var = (img_var - np.min(img_var)) / (np.max(img_var) - np.min(img_var))
    re_img = (re_img - np.min(re_img)) / (np.max(re_img) - np.min(re_img))
    result = Mahalanobis((img_var - re_img))
    sio.savemat('./results/{}-result.mat'.format(file), {'result': result})
    plt.title('{}'.format(file))
    plt.savefig('./results/{}-result.png'.format(file))

    return


if __name__ == "__main__":

    for file in ['abu-beach-1', 'Pavia_100',  'GrandIsle', 'MUUFLGulfport', 'UHAD-U-I', 'UHAD-U-II']:
        main(file)
