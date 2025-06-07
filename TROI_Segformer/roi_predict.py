# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from jsegformer import SegFormer_Segmentation

if __name__ == "__main__":
    detect_files = ['cross01', 'cross02', 'cross03', 'cross04', 'cross05']
    # detect_files = ['cross01']
    label_classes = [2]
    for label_class in label_classes:
        for detect_file in detect_files:
            segformer = SegFormer_Segmentation(f"logs/ours_ti_0220/atten{label_class}/{detect_file}/best_epoch_weights.pth",ai=label_class)
            mode = "dir_predict"
            count = False
            name_classes = ["background", "roi"]

            dir_origin_path = f"VOCdevkit/VOC2007/cross/{detect_file}/test"
            # dir_origin_path = 'img/need'
            dir_save_path = f"out/ours_ti_{label_class}0220"

            if mode == "dir_predict":
                import os
                from tqdm import tqdm

                img_names = os.listdir(dir_origin_path)
                for img_name in tqdm(img_names):
                    if img_name.lower().endswith(
                            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                        image_path = os.path.join(dir_origin_path, img_name)
                        image = Image.open(image_path)
                        r_image = segformer.detect_image(image)
                        if not os.path.exists(dir_save_path):
                            os.makedirs(dir_save_path)
                        r_image.save(os.path.join(dir_save_path, img_name))

