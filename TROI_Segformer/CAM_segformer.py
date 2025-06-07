"""
CAM热图观察模型关注点变化
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
from einops import rearrange, einsum
import copy
import torch
import torch.nn.functional as F
from torch import nn, optim


from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def BGR2RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg


model = SegFormer(num_classes=2, phi='b3', pretrained=False, atten_index=2)
model.cuda().eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('logs/best_epoch_weights.pth', map_location=device))

print(model)
def main(img_path):
    # img_path = '../cam_img/v01_ctd164th.jpg'
    image = Image.open(img_path)
    image = cvtColor(image)
    # ---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    # ---------------------------------------------------#
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (512, 512))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    images = torch.from_numpy(image_data)

    images = images.cuda()
    with torch.no_grad():
         pr = model(images)[0][0]
         pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
         pr = pr[int((512 - nh) // 2): int((512 - nh) // 2 + nh), \
              int((512 - nw) // 2): int((512 - nw) // 2 + nw)]

    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
    pr = pr.argmax(axis=-1)
    seg_img = np.reshape(np.array([ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                                (128, 64, 12)], np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    #------------------------------------------------#
    #   将新图片转换成Image的形式
    #------------------------------------------------#
    pr   = Image.fromarray(np.uint8(seg_img))
    #------------------------------------------------#
    #   将新图与原图及进行混合
    #------------------------------------------------#
    pr   = Image.blend(old_img, pr, 0.5)


    class GradCamBackbone(nn.Module):
         def __init__(self, model, layer):
              super().__init__()
              self.model = model
              self.layer = layer
              self.register_hooks()

         def register_hooks(self):

              # 主干输出
              for module_name, module in self.model._modules.items():
                   if module_name == 'backbone':
                        for layer_name, module1 in module._modules.items():
                             # print(layer_name)
                             if layer_name == self.layer:
                                  module1.register_forward_hook(self.forward_hook)
                                  module1.register_backward_hook(self.backward_hook)

         def forward(self, input, target_index):
              outs = self.model(input)[1]
              outs = outs.squeeze()

              if target_index is None:
                   target_index = torch.argmax(outs, dim=0)

              outs[target_index, 0, 0].backward(retain_graph=True)

              height = np.sqrt(self.forward_result.shape[0]).astype(int)
              self.forward_result = rearrange(self.forward_result, '(h w) c -> c h w', h=height)
              self.backward_result = rearrange(self.backward_result, '(h w) c -> c h w', h=height)
              a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)

              out = torch.sum(a_k * self.forward_result, dim=0).cpu()
              out = torch.relu(out) / torch.max(out)
              out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [512, 512])
              return out.cpu().detach().squeeze().numpy()

         def forward_hook(self, _, input, output):
              self.forward_result = torch.squeeze(output)

         def backward_hook(self, _, grad_input, grad_output):
              self.backward_result = torch.squeeze(grad_output[0])


    class GradCamDan(nn.Module):
        def __init__(self, model, layer):
            super().__init__()
            self.model = model
            self.layer = layer
            self.register_hooks()

        def register_hooks(self):
            for module_name, module in self.model._modules.items():
                if module_name == 'dan_decode_head':
                    for layer_name1, module1 in module._modules.items():
                        # print('-', layer_name1)
                        if layer_name1 == self.layer:
                            if self.layer == 'linear_pred':
                                module1.register_forward_hook(self.forward_hook)
                                module1.register_backward_hook(self.backward_hook)
                            for layer_name2, module2 in module1._modules.items():
                                # print('--', layer_name2)
                                if layer_name2 == 'proj' or layer_name2 == 'conv' :
                                    module2.register_forward_hook(self.forward_hook)
                                    module2.register_backward_hook(self.backward_hook)


        def forward(self, input, target_index):
            outs = self.model(input)[1]
            outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]

            if target_index is None:
                 target_index = torch.argmax(outs, dim=0)

            outs[target_index, 0, 0].backward(retain_graph=True)

            if len(self.forward_result.shape) == 2:
                height = np.sqrt(self.forward_result.shape[0]).astype(int)
                self.forward_result = rearrange(self.forward_result, '(h w) c -> c h w', h=height)
                self.backward_result = rearrange(self.backward_result, '(h w) c -> c h w', h=height)
            # else:
            #     height = np.sqrt(self.forward_result.shape[1]).astype(int)
            #     self.forward_result = rearrange(self.forward_result, 'k (h w) c -> (c k) h w', h=height)
            #     self.backward_result = rearrange(self.backward_result, 'k (h w) c -> (c k) h w', h=height)


            a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)
            out = torch.sum(a_k * self.forward_result, dim=0).cpu()
            out = torch.relu(out) / torch.max(out)
            out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [512, 512])
            return out.cpu().detach().squeeze().numpy()

        def forward_hook(self, _, input, output):
            self.forward_result = torch.squeeze(output)

        def backward_hook(self, _, grad_input, grad_output):
            self.backward_result = torch.squeeze(grad_output[0])


    norms = []
    for layy in ['norm1', 'norm2', 'norm3', 'norm4']:
         grad_cam = GradCamBackbone(model=model, layer=layy)
         input_tensor = images
         mask = grad_cam(input_tensor, 0)

         heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
         heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
         heatmap = heatmap[int((512 - nh) // 2): int((512 - nh) // 2 + nh), \
              int((512 - nw) // 2): int((512 - nw) // 2 + nw)]
         heatmap = cv2.resize(heatmap, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
         cam = np.float32(heatmap)+ np.float32(image)
         cam = cam / np.max(cam)
         # cam = cv2.cvtColor(cam,cv2.COLOR_BGR2RGB)
         norms.append(cam)
         # cv2.imshow('hh', cam)
         # cv2.waitKey()


    dans = []
    for layy in ['linear_pred', 'linear_fuse', 'linear_c2', 'linear_c3','linear_c4']:
         grad_cam = GradCamDan(model=model, layer=layy)
         input_tensor = images
         mask = grad_cam(input_tensor, 0)

         heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
         heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

         heatmap = cv2.resize(heatmap, (512, 512), interpolation = cv2.INTER_LINEAR)
         heatmap = heatmap[int((512 - nh) // 2): int((512 - nh) // 2 + nh), \
              int((512 - nw) // 2): int((512 - nw) // 2 + nw)]
         heatmap = cv2.resize(heatmap, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
         cam = np.float32(heatmap)+ np.float32(image)
         cam = cam / np.max(cam)
         # cam = cv2.cvtColor(cam,cv2.COLOR_BGR2RGB)
         dans.append(cam)



    plt.figure(figsize=(16,5))
    # plt.subplot(2,5,1), plt.imshow(old_img)
    # plt.title(f'input'), plt.axis('off')


    plt.subplot(2,5,1), plt.imshow(pr)
    plt.title(f'predict'), plt.axis('off')

    for index, cam_img in enumerate(norms):
        plt.subplot(2,5,index+2), plt.imshow(cam_img)
        plt.title(f'cam_of_encoder_norm{index+1}'), plt.axis('off')

    dansname = ['linear_pred', 'linear_fuse', 'linear_c2', 'linear_c3', 'linear_c4']
    for index, cam_img in enumerate(dans):
        plt.subplot(2,5,index+6), plt.imshow(cam_img)
        plt.title(f'cam_of_dandecoder_{dansname[index]}'), plt.axis('off')


    plt.tight_layout()
    plt.savefig('cam_dan_myseg_'+img_path.split('/')[-1], dpi=300)


if __name__ == '__main__':
    img_pathes = glob.glob('../cam_img/*.jpg')
    for each in img_pathes:
        main(each)













