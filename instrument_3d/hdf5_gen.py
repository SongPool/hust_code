"""
选取不同背景的blender文件，随机改变器械位姿，并生成对应的HDF5文件，保存器械关键点的图像位置和世界坐标
"""
import argparse
import math
import random

import numpy as np
import blenderproc as bproc


# 全局配置
DATE_TAG = "701"


def setup_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default="dataset_scene/b00.blend", help="Path to the scene blend file")
    parser.add_argument('--output_dir', default=f"render_output/out_{DATE_TAG}/", help="Output directory for rendered files")
    return parser.parse_args()


def set_camera_intrinsics():
    """设置相机内参"""
    bproc.camera.set_intrinsics_from_blender_params(
        lens=10, lens_unit='MILLIMETERS',
        image_width=1920, image_height=1080
    )


def setup_camera_pose():
    """设置初始相机位姿"""
    poi = np.array([0, 0, 0])
    location = np.array([0, -2, 0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def setup_lights():
    """设置光源"""
    light_locations = [[7, -2, 7], [-7, -2, 7], [0, -2, 0], [7, -2, -7], [-7, -2, -7]]
    light_energy = 700

    lights = []
    for loc in light_locations:
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(loc)
        light.set_energy(light_energy)
        light.set_distance(10)
        lights.append(light)
    return lights


def find_objects_by_name(objs):
    """查找并分类场景中的物体对象"""
    obj_map = {
        'left_f00': None,
        'left_f01': None,
        'left_f02': None,
        'left_f03': None,
        'lp0': None,
        'lp1': None,
        'lp2': None,
        'lp3': None,
        'right_f00': None,
        'right_f01': None,
        'right_f02': None,
        'right_f03': None,
        'rp0': None,
        'rp1': None,
        'rp2': None,
        'rp3': None
    }

    for obj in objs:
        name = obj.get_name()
        for key in obj_map:
            if name.count(key) > 0:
                obj_map[key] = obj
                break

    return obj_map


def randomize_left_right_angles(left_f01, left_f02, right_f01, right_f02):
    """随机设置左右手指关节角度"""
    angle_deg = random.uniform(0, 30)
    rad_angle = math.radians(angle_deg)

    # 左手部分
    euler = list(left_f01.get_rotation_euler())
    euler[1] = rad_angle
    left_f01.set_rotation_euler(euler)

    euler = list(left_f02.get_rotation_euler())
    euler[1] = -rad_angle
    left_f02.set_rotation_euler(euler)

    # 右手部分
    angle_deg = random.uniform(0, 30)
    rad_angle = math.radians(angle_deg)

    euler = list(right_f01.get_rotation_euler())
    euler[1] = rad_angle
    right_f01.set_rotation_euler(euler)

    euler = list(right_f02.get_rotation_euler())
    euler[1] = -rad_angle
    right_f02.set_rotation_euler(euler)


def randomize_hand_positions(left_f00, right_f00):
    """随机设置左右手掌位置和旋转"""
    # 设置右手
    right_f00.set_location(np.random.uniform([-10, 0, -4], [0, 3, 4]))
    if random.random() < 0.5:
        right_f00.set_rotation_euler(
            np.random.uniform([-np.pi / 4, np.pi * 13 / 18, 0],
                              [-np.pi / 8, np.pi * 20 / 18, np.pi * 1 / 3]))
    else:
        right_f00.set_rotation_euler(
            np.random.uniform([np.pi / 8, np.pi * 13 / 18, 0],
                              [np.pi / 4, np.pi * 20 / 18, np.pi * 1 / 3]))

    # 设置左手
    left_f00.set_location(np.random.uniform([0, 1, -4], [10, 3, 4]))
    if random.random() < 0.5:
        left_f00.set_rotation_euler(
            np.random.uniform([-np.pi / 4, -np.pi / 3, -np.pi * 1 / 3],
                              [-np.pi / 8, np.pi / 3, 0]))
    else:
        left_f00.set_rotation_euler(
            np.random.uniform([np.pi / 8, -np.pi / 3, -np.pi * 1 / 3],
                              [np.pi / 4, np.pi / 3, 0]))


def collect_keypoint_positions(lp_objs, rp_objs, all_world_points, all_image_points):
    """收集关键点的3D世界坐标及2D图像投影"""
    point_xyzes = []
    for p in lp_objs + rp_objs:
        if p is not None:
            mat = p.get_local2world_mat()
            xyz = np.array([mat[0][-1], mat[1][-1], mat[2][-1]])
            point_xyzes.append(xyz)

    if point_xyzes:
        all_world_points.append(np.array(point_xyzes))
        loc2d = bproc.camera.project_points(np.array(point_xyzes))
        all_image_points.append(loc2d)


def render_and_save(output_path, scene_index, num_images=20):
    """渲染指定数量的图像并保存为HDF5文件"""
    all_world_points = []
    all_image_points = []

    for img_id in range(num_images):
        # 随机变换
        randomize_left_right_angles(obj_map['left_f01'], obj_map['left_f02'],
                                    obj_map['right_f01'], obj_map['right_f02'])
        randomize_hand_positions(obj_map['left_f00'], obj_map['right_f00'])

        # 收集关键点
        collect_keypoint_positions(
            [obj_map['lp0'], obj_map['lp1'], obj_map['lp2'], obj_map['lp3']],
            [obj_map['rp0'], obj_map['rp1'], obj_map['rp2'], obj_map['rp3']],
            all_world_points, all_image_points
        )

        # 渲染
        data = bproc.renderer.render()

        # 保存
        bproc.writer.write_hdf5(f"{output_path}roate{img_id:03d}", data)

    # 保存关键点数据
    np.save(f'render_output/out_{DATE_TAG}/scene{scene_index:02d}_img_loc_{DATE_TAG}.npy', np.array(all_image_points))
    np.save(f'render_output/out_{DATE_TAG}/scene{scene_index:02d}_word_loc_{DATE_TAG}.npy', np.array(all_world_points))


def main():
    args = setup_parser()
    bproc.init()

    set_camera_intrinsics()
    setup_camera_pose()
    setup_lights()

    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    NUM_SCENES = 23
    NUM_IMAGES_PER_SCENE = 20

    for scene_index in range(NUM_SCENES):
        # 更新输入输出路径
        args.scene = f"dataset_scene/b{scene_index:02d}.blend"
        output_dir = f"render_output/out_{DATE_TAG}/scene{scene_index:02d}/"

        # 加载场景对象
        objects = bproc.loader.load_blend(args.scene)
        global obj_map
        obj_map = find_objects_by_name(objects)

        # 开始渲染
        render_and_save(output_dir, scene_index, NUM_IMAGES_PER_SCENE)

        # 隐藏所有物体（准备下一场景）
        for obj in objects:
            obj.hide()


if __name__ == "__main__":
    main()
