
def calculate_slope_and_intercept(point1, point2):
    # 提取点的坐标
    x1, y1 = point1
    x2, y2 = point2

    # 计算斜率
    if x1 == x2:  # 避免除数为0的情况（即垂直线）
        slope = float('inf')  # 或者你可以返回一个错误或特定的值来表示垂直线
        return slope, x1
    else:
        slope = (y2 - y1) / (x2 - x1)

        # 使用一个点的坐标和斜率来计算截距
    intercept = y1 - slope * x1

    return slope, intercept

# 示例用法


def calculate_intersection(m1, b1, m2, b2):
    # 检查两条直线是否平行
    if m1 == m2:
        return None, None  # 平行线没有交点
    elif m1 == float('inf'):
        return b1, m2 * b1 + b2
    elif m2 == float('inf'):
        return b2, m1 * b2 + b1
    # 使用公式计算交点
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return x, y

# 示例用法
#
# point1 = (1, 2)
# point2 = (3, 4)
# slope1, intercept1 = calculate_slope_and_intercept(point1, point2)
# point1 = (1, -2)
# point2 = (3, -4)
# slope2, intercept2 = calculate_slope_and_intercept(point1, point2)
# print(f"Slope: {slope1}, Intercept: {intercept1}")
# print(f"Slope: {slope2}, Intercept: {intercept2}")
#
# intersection = calculate_intersection(slope1, intercept1, slope2, intercept2)
# if intersection is not None:
#     x, y = intersection
#     print(f"The intersection point is ({x}, {y})")
# else:
#     print("The lines are parallel and do not intersect.")


#根据像素坐标反算空间坐标
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])
#
# # 假设的像素坐标 (u, v)
# u, v = 100, 200  # 示例值
# pixel_coords = np.array([[u], [v], [1]])  # 转换为齐次坐标
#
# # 计算归一化摄像机坐标系中的坐标
# normalized_coords = np.dot(np.linalg.inv(K), pixel_coords)
# Xc, Yc, Zc = normalized_coords[0], normalized_coords[1], normalized_coords[2]
#
# # 注意：由于Zc在归一化摄像机坐标系中总是1，所以通常我们不会使用它
# print(f"Xc: {Xc}, Yc: {Yc}")

##todo 光束法平差
# import numpy as np
# from scipy.optimize import minimize
#
# # 假设的内参矩阵
# K = np.array([[1000, 0, 640],
#               [0, 1000, 360],
#               [0, 0, 1]])
#
# # 假设的三维点（世界坐标系）
# P_world = np.array([[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]])

# # 假设的相机外参（旋转矩阵R和平移向量t）
# # 注意：这些应该是优化变量的一部分，但为了示例，我们先假设一些值
# R = np.array([[0.9239, -0.3827, 0.],
#               [0.3827, 0.9239, 0.],
#               [0., 0., 1.]])
# t = np.array([0, 0, 10])
#
# # 假设的图像点（观测值）
# # 这些应该是真实的观测值，但在这里我们只是随机生成一些值
# x_obs, y_obs = np.random.rand(2, len(P_world)) * 800


# 相机投影函数
def project(P_world, R, t, K):
    # 变换到相机坐标系
    P_cam = np.dot(R, (P_world.T - t).T)
    # 归一化坐标
    P_norm = P_cam[:, :3] / P_cam[:, 2]
    # 投影到图像平面
    p_image = np.dot(K, P_norm.T).T
    # 提取x, y坐标
    x, y = p_image[:, 0] / p_image[:, 2], p_image[:, 1] / p_image[:, 2]
    return x, y


# 重投影误差函数（目标函数）
def reprojection_error(params, P_world, x_obs, y_obs, K):
    # 假设params是一个扁平化的数组，包含R和t的元素
    R_flat = params[:9].reshape(3, 3)
    t_flat = params[9:].reshape(3, 1)

    # 投影三维点到图像平面
    x_pred, y_pred = project(P_world, R_flat, t_flat, K)

    # 计算重投影误差
    error = np.sqrt((x_pred - x_obs) ** 2 + (y_pred - y_obs) ** 2)
    return np.sum(error)

# # 初始参数（扁平化）
# init_params = np.concatenate((R.flatten(), t.flatten()))
# # 使用SciPy的minimize函数进行优化
# result = minimize(reprojection_error, init_params, args=(P_world, x_obs, y_obs, K))
# # 输出优化结果
# print("Optimized parameters:", result.x)


