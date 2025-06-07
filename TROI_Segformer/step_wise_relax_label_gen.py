import glob
import time

import numpy
import torch
import torch.nn as nn
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2 as cv



def kernel_w(size):
    alpha = 60/size
    k_size = size
    # k_size = 61
    temp_w = np.zeros((k_size,k_size))
    for i in range (k_size):
        for j in range(k_size):
            # if i + j > 40:
            #     d = max(i,j) -20
            # else:
            #     d = 20 - min(i,j)
            d = math.sqrt(abs(i - (k_size-1)/2)**2 + abs(j - (k_size-1)/2)**2)
            w = 0.5 / (1 + math.e**(alpha*d-4)) + 0.5 / (1 + math.e**(alpha*d-24))
            temp_w[i,j] = w

    return temp_w



def longer_ax(img, points):
    center, axes, angle = cv.fitEllipse(points)

    ll = int(max(axes)/2)

    if ll % 2 == 0:
        ll += 1

    return ll


def get_minpng(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarch = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        points = contours[0]
        for last_c in contours[1:]:
            points = numpy.vstack((points, last_c))

    else:
        points = contours[0]

    center, axes, angle = cv.fitEllipse(points)

    ll = max(int(max(axes) / 10), 5)

    img1 = np.copy(img)
    cv.drawContours(img1, points, -1, (0,0,0), 0)
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    contours, hierarch = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        points = contours[0]
        for last_c in contours[1:]:
            points = numpy.vstack((points, last_c))

    else:
        points = contours[0]

    return img1, points


def get_dpng(img, k_size,w):
    # 传入src为cv2读取的label
    c2 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=k_size, stride=1)
    c2.weight= nn.Parameter(torch.tensor(w,dtype=torch.float))
    c2.bias = nn.Parameter(torch.tensor([0.0]))
    c2 = c2.cuda()
    gray = img[:, :, 1]
    # 增加维度，使得其能够进行卷积操作   三维（c,w,h）, 四维（b,c,w,h）
    png = numpy.expand_dims(gray, 0)
    # 先将label转换成张量，再卷积生成扩散标签并取出
    png = torch.tensor(png, dtype=torch.float).cuda()
    out = c2(png).cpu().detach().numpy()
    # 归一化处理 峰值亮度改成255
    out = out/out.max()*255
    # 裁剪一手
    bbias = int((k_size-1)/2)
    out = np.stack((np.zeros((480, 854)), out[0][bbias:bbias+480, bbias:bbias+854], np.zeros((480, 854))), axis=-1)
    out = np.array(out, np.uint8)
    return out


def ks_png(png, points, w):
    src = torch.from_numpy(png).cuda()
    for each in points:
        # 创建全为0的400*500的图像
        img = torch.zeros((480, 854)).cuda()
        # 指定要插入的位置
        row_start = int(each[0][1] - (len(w)-1)/2)
        col_start = int(each[0][0] - (len(w)-1)/2)

        # 将要插入的张量放置在图像内
        your_tensor = torch.from_numpy(w)*255
        your_tensor_row_start = max(0, -row_start)
        your_tensor_col_start = max(0, -col_start)
        your_tensor_row_end = your_tensor_row_start + min(len(w), 480 - row_start)
        your_tensor_col_end = your_tensor_col_start + min(len(w), 854 - col_start)
        img[max(row_start, 0):min(row_start + len(w), 480), max(col_start, 0):min(col_start + len(w), 854)] =\
            your_tensor[your_tensor_row_start:your_tensor_row_end, your_tensor_col_start:your_tensor_col_end]

        src = torch.max(src, img)

    return src.cpu().detach().numpy()





if __name__ == "__main__":
    boundarys = glob.glob('../roidata/img/label/*pseudo.png')
    pnglist = glob.glob('../roidata/over/bound_png/*.png')
    for each,bb in zip(pnglist, boundarys):
        name = each.split('\\')[-1]

        raw_src = cv.imread(each)
        raw_src[raw_src > 100] = 255
        # raw_src = np.stack((np.zeros((480, 854)), raw_src[:,:,1], np.zeros((480, 854))), axis=-1)
        raw_src = np.array(raw_src, dtype=np.uint8)

        start = time.time()
        ml_src, points = get_minpng(raw_src)
        # print(points)
        size = longer_ax(ml_src, points)
        w = kernel_w(size)

        after_a = ks_png(ml_src[:,:,1], points, w)

        after_a = np.stack((np.zeros((480, 854)), after_a, np.zeros((480, 854))), axis=-1)
        after_a = numpy.array(after_a, dtype=numpy.uint8)

        b_img = cv.imread(bb)

        b_img[b_img > 0] = 1
        # out = after_a * b_img
        out = after_a

        cv.imwrite('../roidata/'+name, out)
        print('get_one:', time.time() - start)