#    @author Created by Genius_Tian

#    @Date 2021/7/28

#    @Description Canny算子
import math

import cv2
import numpy as np


def Canny(img, sigma, threshold1, threshold2):
    # 确定高斯核大小
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 != 0:
        dim += 1
    # 高斯核填充
    Gaussian_filter = np.zeros([dim, dim])
    for y in range(dim):
        for x in range(dim):
            Gaussian_filter[y, x] = 1 / (2 * math.pi * sigma ** 2) * math.exp(- (y ** 2 + x ** 2) / (2 * sigma ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    #     图像填充
    p = int((dim - 1) / 2)
    h, w = img.shape
    img_pad = np.pad(img, (p, p), (p, p), 'constant')
    img_new = np.zeros(img.shape)
    for dst_h in range(h):
        for dst_w in range(w):
            img_new[dst_h, dst_w] = np.sum(img_pad[dst_h:dst_h + dim, dst_w:dst_w + dim] * Gaussian_filter)
