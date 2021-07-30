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

    #     使用sobel算子检测边缘
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_gradient_x = np.zeros(img_new.shape)
    img_gradient_y = np.zeros(img_new.shape)
    img_gradient = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(h):
        for j in range(w):
            img_gradient_x[i, j] = np.sum(img_pad[i + 3, j + 3] * sobel_x)
            img_gradient_y[i, j] = np.sum(img_pad[i + 3, j + 3] * sobel_y)
            img_gradient[i, j] = np.sqrt(img_gradient_x[i, j] ** 2 + img_gradient_y ** 2)


if __name__ == '__main__':
    a = np.array([[255, 255], [255, 255]])
    b = np.array([[1, 2], [3, 4]])
    zeros = np.zeros(a.shape)
    zeros += np.sum(a * b)
    print(zeros.astype(np.uint8))
    print(a[:1, :])
