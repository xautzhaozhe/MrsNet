import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from PIL import Image
from einops import rearrange


def standard(x):
    max_value = np.max(x)
    min_value = np.min(x)
    if max_value == min_value:
        return np.zeros_like(x)
    return (x - min_value) / (max_value - min_value)


def cosin_similarity(x, y):
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y_multi = np.sum(np.multiply(x, y), axis=1)
    return x_y_multi / (x_norm * y_norm + 1e-8)


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


def Get_Spatial_mask(image, spatial_mask_rate):
    # 读取图像并转为灰度
    arr = np.array(image)
    # 展平数组以便处理
    flatten_arr = arr.flatten()
    n_pixels = flatten_arr.size
    k = int(n_pixels * spatial_mask_rate)

    # 使用 argpartition 高效找到最大k个像素的索引
    # 参数 -k 表示找到最大的k个元素的位置
    indices = np.argpartition(flatten_arr, -k)[-k:]
    # 创建全1数组，将最大k个位置设为0
    binary_flatten = np.ones(n_pixels, dtype=np.uint8)
    binary_flatten[indices] = 0
    # 重塑为原始图像形状
    mask_image = binary_flatten.reshape(arr.shape)

    return mask_image


def spectral_band_smoothing(image, window_size=7):
    # 验证窗口大小是否为奇数
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须是奇数")

    H, W, C = image.shape
    half_window = window_size // 2
    smoothed = np.zeros_like(image, dtype=np.float32)

    # 转换为浮点型确保计算精度
    image = image.astype(np.float32)
    for c in range(C):
        # 计算有效窗口范围
        start = max(0, c - half_window)
        end = min(C, c + half_window + 1)
        # 计算邻域均值
        smoothed[:, :, c] = np.mean(image[:, :, start:end], axis=2)

    return smoothed.astype(image.dtype)  # 保持原始数据类型


def cosine_reconstruction_block(hsi_block, win_size):

    b, _, c = hsi_block.shape
    outer_radius = b // 2  # 外窗口半径是块大小的一半
    inner_radius = win_size // 2
    # 获取中心像素
    center = hsi_block[outer_radius, outer_radius, :]
    # 初始化权重和重构像素
    weights = []
    neighbors = []

    # 遍历外窗口内的所有像素
    for i in range(b):
        for j in range(b):
            # 跳过内窗口内的像素
            if abs(i - outer_radius) <= inner_radius and abs(j - outer_radius) <= inner_radius:
                continue
            # 获取当前像素
            pixel = hsi_block[i, j, :]

            # 计算余弦相似度
            dot_product = np.dot(center, pixel)
            norm_center = np.linalg.norm(center)
            norm_pixel = np.linalg.norm(pixel)
            cosine_sim = dot_product / (norm_center * norm_pixel + 1e-12)  # 防止除以零

            # 保存权重和像素
            weights.append(cosine_sim)
            neighbors.append(pixel)

    # 将权重和像素转换为 NumPy 数组
    weights = np.array(weights)
    neighbors = np.array(neighbors)

    # 加权求和重构中心像素
    sum_weights = np.sum(np.abs(weights))  # 使用绝对值保证稳定性
    weights = weights / sum_weights
    weighted_sum = np.sum(weights[:, None] * neighbors, axis=0)

    if sum_weights > 1e-12:
        reconstructed_center = weighted_sum / sum_weights
    else:
        reconstructed_center = center  # 如果权重和为零，保留原始中心像素
    reconstructed_center = reconstructed_center.reshape(1, c)
    return reconstructed_center


def patch_encoded(patch, center):
    p_h, p_w, b = patch.shape
    patch_unfold = np.reshape(patch, [-1, b], order='F')
    assert patch_unfold.shape[1] == center.shape[1]
    encoded_weight = cosin_similarity(patch_unfold, center)
    encoded_weight = np.exp(encoded_weight) / np.sum(np.exp(encoded_weight))
    encoded_weight = encoded_weight[:, None]
    encoded_vector = np.sum(encoded_weight * patch_unfold, axis=0)
    encoded_vector = encoded_vector[None, :]
    return encoded_vector


class Data(data.Dataset):
    def __init__(self, img, wout_size=5, spatial=True, spatial_mask_rate=0.2,
                 spectral=True, spectral_window=7):
        self.w_size = wout_size
        self.pad_size = wout_size // 2
        self.spatial_mask_rate, self.spectral_window = spatial_mask_rate, spectral_window
        self.spatial, self.spectral = spatial, spectral
        self.h, self.w, self.b = img.shape
        self.nums = self.h * self.w
        img = standard(img)

        self.data = np.pad(img, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)),
                           mode='reflect')

        # 空间掩码
        self.mask_data = self.data
        hh, ww, cc = self.data.shape
        self.spa_mask_out, self.mask_out = np.zeros(shape=(hh, ww)), np.zeros(shape=(hh, ww))
        if self.spatial:
            rx_result = Mahalanobis(self.mask_data)
            spa_mask = Get_Spatial_mask(rx_result, spatial_mask_rate)
            spa_mask = np.expand_dims(spa_mask, 2).repeat(self.b, axis=2)
            self.spa_mask_out = self.mask_data * spa_mask

        # 光谱掩码
        if self.spectral:
            self.mask_out = spectral_band_smoothing(self.spa_mask_out, window_size=self.spectral_window)

    def __getitem__(self, index):
        position_y = index // self.w
        position_x = index - position_y * self.w
        position_x = position_x + self.pad_size
        position_y = position_y + self.pad_size

        windows_out = self.data[position_y - self.pad_size:position_y + self.pad_size + 1,
                      position_x - self.pad_size:position_x + self.pad_size + 1, :]

        # 获取中心编码向量
        patch = windows_out
        center = windows_out[self.pad_size, self.pad_size]
        center = center[None, :]
        coded_vector = patch_encoded(patch, center)
        mask_windows_out = self.mask_out[position_y - self.pad_size:position_y + self.pad_size + 1,
                           position_x - self.pad_size:position_x + self.pad_size + 1, :]

        return (torch.unsqueeze(torch.from_numpy(coded_vector).float().permute(1, 0), dim=2),
                torch.from_numpy(windows_out).float().permute(2, 0, 1),
                torch.from_numpy(mask_windows_out).float().permute(2, 0, 1))

    def __len__(self):
        return self.nums

