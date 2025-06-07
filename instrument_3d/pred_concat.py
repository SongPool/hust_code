"""
将二维关键点检测结果和深度估计结果拼接，并处理异常值
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mpl_toolkits.mplot3d


def get_hidedepth(p1,p2, image):
    # 有优化空间，直接算点的距离，但是要分类，先放着
    img = np.array(np.zeros((1400, 2500)), dtype=np.uint8)
    img[160:160+1080, 290:290+1920] = image
    x1, y1 = p1
    x2, y2 = p2
    x1, x2 = x1 + 290, x2 + 290
    y1, y2 = y1 + 160, y2 +160
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 创建一个数组来存储直线上的像素值
    line_values = np.zeros(int(distance))

    # 使用线性插值获取直线上的像素值
    for i in range(int(distance)):
        # 计算当前点在直线上的位置比例
        t = i / distance
        # 使用线性插值公式计算当前点的坐标
        x = int((1 - t) * x1 + t * x2)
        y = int((1 - t) * y1 + t * y2)
        # 获取当前点的像素值并存储到数组中
        line_values[i] = img[y, x]

    # 输出直线上的像素值
    bound_id = 10
    for e in range(len(line_values) - 1, 0, -1):
        if line_values[e] >150:
            bound_id = e
            break

    # 拟合直线
    coff = np.polyfit(np.arange(bound_id), line_values[:bound_id], deg=1)
    hide_d = coff[0]*len(line_values) + coff[1]
    return hide_d


def get3d_xyz(points, depth):
    """
    [p0,p1,p2,p3]

    return [x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]

    """
    tool_center = (points[0] * 2500 - 290, points[1] * 1400 - 160)
    depth[depth<15] = 0

    x3d = []
    y3d = []
    z3d = []
    for i in range(4):
        x = points[i*2] * 2500 - 290
        y = points[i*2+1] * 1400 - 160
        if x > 1920 or x < 0 or y >1080 or y < 0:
            if i == 0:  # 器械中心点在图像外
                return [[-1,-1,-1], [-1,-1,-1], [-1,-1,-1]]

            # 其它点在边界以外的话;
            hide_d = get_hidedepth(tool_center, (x,y), depth)
            z = int(hide_d*2)

        else:
            z = depth[int(y), int(x)] * 2
            if z <50:
                # print(x,y,z)
                # temp = cv2.circle(np.stack([depth, depth, depth], -1), (int(x), int(y)), 4,(255,255,0))
                # cv2.imshow('ss', temp)
                # cv2.waitKey()
                z = int(get_hidedepth(tool_center, (x,y), depth))*2

        # 三维显示只能是正方形，都移动到1920尺度上
        x3d.append(x)
        y3d.append(y + 420)
        z3d.append(z + 705)

    return [x3d, y3d, z3d]


def corr_depth(points_data, threshold = 60):
    """
       对器械尖端深度突变超过阈值的关键点深度进行修正
    """
    for ii in range(1, len(points_data)):
        com = points_data[ii, 2, 0] - points_data[ii - 1, 2, 0]
        err1 = points_data[ii, 2, 1] - points_data[ii - 1, 2, 1]
        err2 = points_data[ii, 2, 2] - points_data[ii - 1, 2, 2]

        if abs(err1 - com) > threshold:
            points_data[ii, 2, 1] = points_data[ii - 1, 2, 1] + com
            change1.append([ii, points_data[ii, 2, 1]])

        if abs(err2 - com) > threshold:
            points_data[ii, 2, 2] = points_data[ii - 1, 2, 2] + com
            change2.append([ii, points_data[ii, 2, 2]])

    return points_data


if __name__ == '__main__':
    left_tool = []
    right_tool = []

    # 获取预测位置并补齐缺失深度
    for png_i in range(1, 141):
        pose_data = np.loadtxt(f'super_pose/labels/big_new_action_{png_i}.txt', dtype=np.float16)
        depth_img = cv2.imread(f'out/wb/{png_i:04d}.png')[:, :, 0]
        # 默认第一个是右器械
        right_i, left_i = 0, 1
        if pose_data[0][5] < pose_data[1][5]:
            right_i, left_i = 1, 0
        right_pose_points = []
        left_pose_points = []
        for pos_id in [5, 6, 8, 9, 11, 12, 14, 15]:
            right_pose_points.append(pose_data[right_i][pos_id])
            left_pose_points.append(pose_data[left_i][pos_id])

        left_tool.append(get3d_xyz(left_pose_points, depth_img))
        right_tool.append(get3d_xyz(right_pose_points, depth_img))

    # 滤波处理异常值
    left_tool = corr_depth(np.array(left_tool))
    right_tool = corr_depth(np.array(right_tool))

    np.save('right_tool_new.npy', right_tool)
    np.save('left_tool_new.npy', left_tool)




