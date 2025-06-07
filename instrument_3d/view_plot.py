"""
预测结果的可视化和误差计算
"""
import sys
import cv2
import numpy as np
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter


metadata = dict(title='Rotating', artist='matplotlib', comment='Rotating Video')
writer = PillowWriter(fps=10, metadata=metadata)

def image_to_world(u, v, Z,
                   K=np.array([[533.33, 0, 960],[0, 450, 540], [0, 0, 1]]),
                   R=np.array([[1., 0, 0],[0, 0, -1], [0, 1, 0]]),
                   T=np.array([[0],[-2], [0]])
                   ):
    """
    将图像坐标 (u, v) 和深度 Z 转换为世界坐标系中的实际坐标。

    参数：
    u, v       : 图像坐标 (像素)
    Z          : 深度（相机坐标系中的 Z 坐标）
    K          : 相机内参矩阵 (3x3)
    R          : 相机旋转矩阵 (3x3), 默认是单位矩阵
    T          : 相机平移向量 (3x1), 默认是零向量

    返回：
    world_coords : 世界坐标系中的坐标 (X_w, Y_w, Z_w)
    """

    # Step 1: 将图像坐标 (u, v) 转换为相机坐标系下的 (X_c, Y_c, Z_c)
    # 计算相机坐标系中的 (X_c, Y_c)
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    # 相机坐标系中的 X_c 和 Y_c
    X_c = (u - c_x) * Z / f_x
    Y_c = (v - c_y) * Z / f_y
    Z_c = Z  # 深度就是相机坐标系中的 Z 坐标

    # Step 2: 将相机坐标系中的 (X_c, Y_c, Z_c) 转换为世界坐标系中的 (X_w, Y_w, Z_w)
    camera_coords = np.array([X_c, Y_c, Z_c]).reshape(3, 1)

    # 使用相机外参转换到世界坐标系
    world_coords = R @ camera_coords + T

    return world_coords.reshape(3)


def get_plot(Xes, Yes, Zes):
    """
    4 points
    [0,1,2,3]
    skl  3-0  0-1 0-2
    """
    X3es = [[Xes[3], Xes[0]], [Xes[0], Xes[1]], [Xes[0], Xes[2]]]
    Y3es = [[Yes[3], Yes[0]], [Yes[0], Yes[1]], [Yes[0], Yes[2]]]
    Z3es = [[Zes[3], Zes[0]], [Zes[0], Zes[1]], [Zes[0], Zes[2]]]

    return X3es, Y3es, Z3es


def img_2_world():
    leftdata = np.load('left_tool_new.npy')
    rightdata = np.load('right_tool_new.npy')

    right_world = []
    for each in rightdata:
        each[2] = (each[2] - 705)/2/255*7
        each[1] = each[1] - 420
        new_e = []
        for ee in range(4):
            new_p = image_to_world(each[0][ee], each[1][ee], each[2][ee])
            new_e.append(new_p)
        right_world.append(new_e)

    np.save('right_tool_world.npy', np.array(right_world))

    left_world = []
    for each in leftdata:
        each[2] = (each[2] - 705)/2/255*7
        each[1] = each[1] - 420
        new_e = []
        for ee in range(4):
            new_p = image_to_world(each[0][ee], each[1][ee], each[2][ee])
            new_e.append(new_p)
        left_world.append(new_e)

    np.save('left_tool_world.npy', np.array(left_world))


def world_gt(n=10):
    all_datas = []
    for points in ['lp0', 'lp1', 'lp2', 'lp3', 'rp0', 'rp1', 'rp2', 'rp3']:
        with open(f"action_data/{points}_positions.txt") as f:
            data = f.readlines()
            temp_loc = []
            for ee in data:
                # print(ee)
                wx = float(ee.split(';')[0].split(',')[0])
                wy = float(ee.split(';')[0].split(',')[1])
                wz = float(ee.split(';')[0].split(',')[2])
                temp_loc.append([wx, wy, wz])
            all_datas.append(copy.copy(temp_loc))

    all_datas = np.array(all_datas)
    left_data = all_datas[:4]
    right_data = all_datas[4:]

    # plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.tight_layout()
    ax.set_aspect('auto')
    # with writer.saving(fig, 'out1_world.gif', 200):
    for ii in range(140):
        plt.cla()
        # ax.set_title('3D vis')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        ax.set_xlim3d([-n, n])
        ax.set_zlim3d([-n, n])
        ax.set_ylim3d([-n,n])
        ax.set_xlabel("X")
        ax.set_zlabel("Y")
        ax.set_ylabel("depth")
        ax.view_init(elev=11, azim=-80)

        # left
        xs = left_data[:,ii,0]
        ys = left_data[:,ii,1]
        zs = left_data[:,ii,2]
        zs,ys = ys, zs

        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='blue', linewidth=4)

        xs = right_data[:,ii,0]
        ys = right_data[:,ii,1]
        zs = right_data[:,ii,2]
        zs, ys = ys, zs
        ax.scatter(xs, zs, ys, marker='o', c='gray', s=50)
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='orange', linewidth=4)

        plt.savefig(f'visss/gt/gt1213_{ii:03d}.png', dpi=200)

        # plt.draw()
        # writer.grab_frame()
        # plt.pause(0.01)

        # plt.show()

def world_pred(n=10):

    left_data = np.load('left_tool_world.npy')
    right_data = np.load('right_tool_world.npy')

    # plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.tight_layout()
    ax.set_aspect('auto')
    # with writer.saving(fig, 'out1_world.gif', 200):
    for ii in range(140):
        plt.cla()
        # ax.set_title('3D vis')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        ax.set_xlim3d([-n, n])
        ax.set_zlim3d([-n, n])
        ax.set_ylim3d([-n,n])
        ax.set_xlabel("X")
        ax.set_zlabel("Y")
        ax.set_ylabel("depth")
        ax.view_init(elev=11, azim=-80)

        # left
        xs = left_data[ii,:,0]
        ys = -left_data[ii,:,1]
        zs = -left_data[ii,:,2]
        zs,ys = ys, zs

        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='orange', linewidth=4)

        xs = right_data[ii,:,0]
        ys = -right_data[ii,:,1]
        zs = -right_data[ii,:,2]
        zs, ys = ys, zs
        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='blue', linewidth=4)

        # plt.show()
        # # plt.draw()
        # # plt.pause(0.01)
        plt.savefig(f'visss/pred/pred1213_{ii:03d}.png', dpi=200)


def world_all(n=10, compensate=True):
    all_datas = []
    for points in ['lp0', 'lp1', 'lp2', 'lp3', 'rp0', 'rp1', 'rp2', 'rp3']:
        with open(f"action_data/{points}_positions.txt") as f:
            data = f.readlines()
            temp_loc = []
            for ee in data:
                wx = float(ee.split(';')[0].split(',')[0])
                wy = float(ee.split(';')[0].split(',')[1])
                wz = float(ee.split(';')[0].split(',')[2])
                temp_loc.append([wx, wy, wz])
            all_datas.append(copy.copy(temp_loc))

    all_datas = np.array(all_datas)
    gt_left_data = all_datas[4:]
    gt_right_data = all_datas[:4]


    left_data = np.load('left_tool_world.npy')
    right_data = np.load('right_tool_world.npy')

    # plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.tight_layout()
    ax.set_aspect('auto')
    # with writer.saving(fig, 'out1_world.gif', 200):
    for ii in range(60,140):
        plt.cla()
        # ax.set_title('3D vis')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        ax.set_xlim3d([-n, n])
        ax.set_zlim3d([-n, n])
        ax.set_ylim3d([-n,n])
        ax.set_xlabel("X")
        ax.set_zlabel("z")
        ax.set_ylabel("y")
        ax.view_init(elev=11, azim=-80)


        # gt left
        xs = gt_left_data[:, ii, 0]
        ys = gt_left_data[:, ii, 1]
        zs = gt_left_data[:, ii, 2]
        zs, ys = ys, zs
        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='orange', linewidth=4, alpha=0.3)

        # gt right
        xs = gt_right_data[:, ii, 0]
        ys = gt_right_data[:, ii, 1]
        zs = gt_right_data[:, ii, 2]
        zs, ys = ys, zs
        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='blue', linewidth=4, alpha=0.3)


        if compensate:
            compensate_x = gt_left_data[3,ii,0] - left_data[ii,3,0]
            compensate_y = gt_left_data[3,ii,2] - -left_data[ii,3,2]
            compensate_z = gt_left_data[3,ii,1] - -left_data[ii,3,1]


        # pred left
        xs = left_data[ii,:,0]
        ys = -left_data[ii,:,1]
        zs = -left_data[ii,:,2]
        zs,ys = ys, zs

        if compensate:
            xs = xs + compensate_x
            ys = ys + compensate_y
            zs = zs + compensate_z


        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='orange', linewidth=4)
        # pred right

        xs = right_data[ii,:,0]
        ys = -right_data[ii,:,1]
        zs = -right_data[ii,:,2]
        zs, ys = ys, zs
        if compensate:
            xs = xs + compensate_x
            ys = ys + compensate_y
            zs = zs + compensate_z

        ax.scatter(xs, zs, ys, marker='o', c='gray')
        sklxs, sklys, sklzs = get_plot(xs, zs, ys)
        for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            ax.plot(exs, eys, ezs, c='blue', linewidth=4)

        plt.show()
        # # plt.draw()
        # # plt.pause(0.01)
        # plt.savefig(f'visss/sum/sum1213_{ii:03d}.png', dpi=200)


def world_all_mmpose(n=10, compensate=False):
    all_datas = []
    for points in ['lp0', 'lp1', 'lp2', 'lp3', 'rp0', 'rp1', 'rp2', 'rp3']:
        with open(f"action_data/{points}_positions.txt") as f:
            data = f.readlines()
            temp_loc = []
            for ee in data:
                # print(ee)
                wx = float(ee.split(';')[0].split(',')[0])
                wy = float(ee.split(';')[0].split(',')[1])
                wz = float(ee.split(';')[0].split(',')[2])
                temp_loc.append([wx, wy, wz])
            all_datas.append(copy.copy(temp_loc))

    all_datas = np.array(all_datas)
    gt_left_data = all_datas[4:]
    gt_right_data = all_datas[:4]


    left_data = np.load('world_pred1_mmpose.npy')
    # right_data = np.load('right_tool_world.npy')

    # plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.tight_layout()
    ax.set_aspect('auto')
    with writer.saving(fig, 'out1_world_mmpose.gif', 200):
        for ii in range(140):
            plt.cla()
            # ax.set_title('3D vis')
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_zticklabels([])
            ax.set_xlim3d([-n, n])
            ax.set_zlim3d([-n, n])
            ax.set_ylim3d([-n,n])
            ax.set_xlabel("X")
            ax.set_zlabel("z")
            ax.set_ylabel("y")
            ax.view_init(elev=11, azim=-80)


            # # gt left
            # xs = gt_left_data[:, ii, 0]
            # ys = gt_left_data[:, ii, 1]
            # zs = gt_left_data[:, ii, 2]
            # zs, ys = ys, zs
            # ax.scatter(xs, zs, ys, marker='o', c='gray')
            # sklxs, sklys, sklzs = get_plot(xs, zs, ys)
            # for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            #     ax.plot(exs, eys, ezs, c='orange', linewidth=4, alpha=0.3)
            #
            # # gt right
            # xs = gt_right_data[:, ii, 0]
            # ys = gt_right_data[:, ii, 1]
            # zs = gt_right_data[:, ii, 2]
            # zs, ys = ys, zs
            # ax.scatter(xs, zs, ys, marker='o', c='gray')
            # sklxs, sklys, sklzs = get_plot(xs, zs, ys)
            # for exs, eys, ezs in zip(sklxs, sklys, sklzs):
            #     ax.plot(exs, eys, ezs, c='blue', linewidth=4, alpha=0.3)


            if compensate:
                compensate_x = gt_left_data[3,ii,0] - left_data[ii,3,0]
                compensate_y = gt_left_data[3,ii,2] - -left_data[ii,3,2]
                compensate_z = gt_left_data[3,ii,1] - -left_data[ii,3,1]


            # pred left
            xs = left_data[ii,:4,0]
            ys = left_data[ii,:4,1]
            zs = left_data[ii,:4,2]
            zs,ys = ys, zs

            if compensate:
                xs = xs + compensate_x
                ys = ys + compensate_y
                zs = zs + compensate_z


            ax.scatter(xs, zs, ys, marker='o', c='gray')
            sklxs, sklys, sklzs = get_plot(xs, zs, ys)
            for exs, eys, ezs in zip(sklxs, sklys, sklzs):
                ax.plot(exs, eys, ezs, c='blue', linewidth=4)
            # pred right

            xs = left_data[ii,4:,0]
            ys = left_data[ii,4:,1]
            zs = left_data[ii,4:,2]
            zs, ys = ys, zs
            if compensate:
                xs = xs + compensate_x
                ys = ys + compensate_y
                zs = zs + compensate_z

            ax.scatter(xs, zs, ys, marker='o', c='gray')
            sklxs, sklys, sklzs = get_plot(xs, zs, ys)
            for exs, eys, ezs in zip(sklxs, sklys, sklzs):
                ax.plot(exs, eys, ezs, c='orange', linewidth=4)

        # plt.show()
            plt.draw()
            plt.pause(0.01)
            # plt.savefig(f'visss/sum/sum1213_{ii:03d}.png', dpi=200)



if __name__ == '__main__':
    # world_gt(8)
    # # img_2_world()
    # world_pred(8)

    # world_all(8)
    world_all_mmpose()









