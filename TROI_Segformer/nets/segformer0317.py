# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
# from backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
import matplotlib.pyplot as plt


def show_feature_map(feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(512, 512))  # 这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
    plt.show()


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MyAttention(nn.Module):
    def __init__(self, in_channels):
        super(MyAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # 计算通道维度的平均值，用于计算注意力权重
        avg = self.avgpool(x)
        mask = mask.repeat(1, 64, 1,1)
        # 通过卷积和标准化生成注意力权重
        out = self.conv(avg)
        out = self.bn(out)
        out = self.sigmoid(out)
        # 使用注意力权重加权编码器输出的特征
        out = x * out
        # 使用掩码增强胆囊预测结果
        out = out * mask
        return out


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*3,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        _c = self.dropout(_c)
        x = self.linear_pred(_c)
        return x


class SegFormerHeadP(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[21, 32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHeadP, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c0_in_channels  = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c0 = MLP(input_dim=c0_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4, c0 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c0 = self.linear_c0(c0).permute(0, 2, 1).reshape(n, -1, c0.shape[2], c0.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c0], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b3', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHeadP(num_classes, [64, 128, 320, 512, 64], self.embedding_dim)
        self.dan_decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        self.fusion = MyAttention(self.in_channels[0])

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        dan = self.dan_decode_head.forward(x)
        fus = self.fusion(x[0], dan[:, 1:])

        x.append(fus)
        x = self.decode_head.forward(x)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        dan = F.interpolate(dan, size=(H, W), mode='bilinear', align_corners=True)

        return dan, x


from PIL import Image
import numpy as np


def preprocess_input(image):
    image /= 255.0
    return image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:

        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


if __name__ == '__main__':
    re = SegFormer(num_classes=2)
    # re.load_state_dict(torch.load('../logs/best_epoch_weights.pth'))
    re.cuda()
    #
    image = Image.open('../../test_img/v01_ctd001th.jpg')

    image_data, nw, nh = resize_image(image, (512, 512))

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    re.eval()

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        dan, x = re(images)

