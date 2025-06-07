"""
从渲染好的hdf5文件中获取rgb和深度图像
"""
import random
import h5py
import numpy as np
import cv2
import tqdm
from pathlib import Path

def save_image(img, scene, img_id, output_dir, suffix=""):
    """保存图像到指定目录"""
    filename = output_dir / f"scene{scene:02d}_{img_id:03d}th{suffix}.jpg"
    cv2.imwrite(str(filename), img)

def main():
    train_list = []
    test_list = []
    min_depth = float('inf')
    max_depth = 0.0

    base_path = Path("render_output/out_701")
    output_img_dir = Path("dataset/img0701")
    output_depth_dir = Path("dataset/depth")

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_depth_dir.mkdir(parents=True, exist_ok=True)

    for scene in range(1, 23):
        for img_id in tqdm.tqdm(range(20), desc=f"Processing Scene {scene:02d}"):
            scene_dir = base_path / f"scene{scene:02d}" / f"roate{img_id:03d}"
            hdf5_file = scene_dir / "0.hdf5"

            try:
                data = h5py.File(str(hdf5_file), 'r')
            except FileNotFoundError:
                print(f"File not found: {hdf5_file}")
                continue

            # 处理颜色图像
            raw = np.array(data['colors'])
            raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            save_image(raw, scene, img_id, output_img_dir, suffix="0701")

            # 处理深度图（可选）
            if 'depth' in data:
                depth = np.array(data['depth'])
                min_depth = min(min_depth, np.min(depth))
                max_depth_val = np.max(depth)

                depth[depth > 6.95] = 0
                max_depth = max(max_depth, max_depth_val)

                # 归一化并保存
                depth_normalized = np.clip(depth * 255 / 7, 1, 255).astype(np.uint8)
                depth_filename = output_depth_dir / f"scene{scene:02d}_{img_id:03d}th.png"
                cv2.imwrite(str(depth_filename), depth_normalized)

                # 数据集划分
                img_name = f"data/img/scene{scene:02d}_{img_id:03d}th.jpg"
                depth_name = f"data/depth/scene{scene:02d}_{img_id:03d}th.png"
                line = f"{img_name} {depth_name} 100"

                if random.random() < 0.1:
                    test_list.append(line)
                else:
                    train_list.append(line)

    print(f"Max depth: {max_depth}")
    print(f"Min depth: {min_depth}")

    # 写入训练与测试列表文件
    with open("depth_train.txt", "w") as f:
        f.write("\n".join(train_list))

    with open("depth_test.txt", "w") as f:
        f.write("\n".join(test_list))

if __name__ == "__main__":
    main()

