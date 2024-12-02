import time
from glob import glob
import json
import numpy as np
import cv2
import os
import utils

import cv2
import numpy as np
from itertools import combinations

import json

# # 假设已知数据
# camera_matrix = np.array([...])  # 摄像机内参矩阵
# images = [...]  # 所有图像路径列表
# R_list = [...]  # 每个图像的旋转矩阵列表
# t_list = [...]  # 每个图像的平移向量列表
# matches = {...}  # 匹配信息字典，键为图像对，值为匹配点坐标对列表


# # 三角化过程，遍历所有可能的图像对进行三角化
# for (img1_idx, img2_idx) in combinations(range(len(images)), 2):
#     # 获取当前图像对的匹配点
#     kp_pairs = matches[(img1_idx, img2_idx)]
#
#     # 提取匹配点坐标
#     points2d_1 = np.float32([kp_pairs[i][0].pt for i in range(len(kp_pairs))])
#     points2d_2 = np.float32([kp_pairs[i][1].pt for i in range(len(kp_pairs))])
#
#     # 构建投影矩阵P1和P2
#     P1 = np.dot(camera_matrix, np.hstack((R_list[img1_idx], t_list[img1_idx])))
#     P2 = np.dot(camera_matrix, np.hstack((R_list[img2_idx], t_list[img2_idx])))
#
#     # 三角化
#     points4D = cv2.triangulatePoints(P1, P2, points2d_1.T, points2d_2.T)
#     points3D = cv2.convertPointsFromHomogeneous(points4D.T)
#
#     # 添加到点云中，注意去重复点
#     for point in points3D:
#         if not any(np.allclose(point, pc, atol=1e-4) for pc in point_cloud):
#             point_cloud.append(point)
#
# # 将稀疏点云转换为NumPy数组
# point_cloud = np.array(point_cloud)
#
# print(f"构建的稀疏点云包含 {len(point_cloud)} 个点")



def cal_intri_matrix(v_x, v_y, v_z, h, w):

    v_x = v_x / np.linalg.norm(v_x)
    v_y = v_y / np.linalg.norm(v_y)
    v_z = v_z / np.linalg.norm(v_z)

    image_width, image_height = h, w
    c_x, c_y = image_width / 2, image_height / 2

    f_x = np.linalg.norm(v_x - [c_x, c_y])
    f_y = np.linalg.norm(v_y - [c_x, c_y])

    # 构建内参矩阵K
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])

    return K

def get_intriMetrix():
    intriMetrix = {}
    lind_idx_combination = [(1, 2), (1, 3), (2, 3)]
    letters = ["HH", "LL", "RR"]
    pics_path = r"calibration/pics"
    json_path = r"calibration/jsons"
    each_dir_metrix = {str(x):None for x in range(54)}
    for file in list(os.walk(json_path))[0][2]:
        avlid_data = {}
        f = open(json_path+"/"+file,"r",encoding="utf8")
        img = cv2.imread(pics_path + "/" + file.split(".")[0] + ".jpg",1)
        h,w,c = img.shape
        data = json.load(f)
        for j in data["shapes"]:
            avlid_data[j["label"]] = j["points"]
        vanish_points = {}
        for letter in letters:
            vanish_point_temp = []
            for idx in lind_idx_combination:
                points_line_one = avlid_data[letter + str(idx[0])]
                points_line_two = avlid_data[letter + str(idx[1])]
                line1_slope, line1_bias = utils.calculate_slope_and_intercept(points_line_one[0], points_line_one[1])
                line2_slope, line2_bias = utils.calculate_slope_and_intercept(points_line_two[0], points_line_two[1])
                vanish_point = utils.calculate_intersection(line1_slope, line1_bias, line2_slope, line2_bias)
                if not np.isnan(vanish_point[0]):
                    vanish_point_temp.append(list(vanish_point))
            vanish_points[letter] = vanish_point_temp
        #print(vanish_points)
        if "A" in file:
            file_name_list = file.split(".")[0].split("A")
        elif "-" in file:
            temp = file.split(".")[0].split("-")
            file_name_list = [str(x) for x in range(int(temp[0]),int(temp[1])+1)]
        else:
            file_name_list = file.split(".")[0]

        v_x = np.mean(vanish_points["LL"],axis=0)
        v_y = np.mean(vanish_points["RR"], axis=0)
        v_z = np.mean(vanish_points["HH"], axis=0)

        K = cal_intri_matrix(v_x,v_y,v_z, h, w)
        for dix in file_name_list:
            intriMetrix[dix] = K
    return intriMetrix


def match_keypoints_sift(img1, img2, place_cut_img, dir_id):
    # 加载图像

    img1 = img1[place_cut_img[str(dir_id)]:, :1000]
    img2 = img2[place_cut_img[str(dir_id)]:, :1000]

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()  # 注意：根据你的OpenCV版本，这行代码可能需要调整

    # 找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT通常使用L2范数进行匹配

    # 匹配描述符
    matches = bf.match(des1, des2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) == 0:
        return [],[]

    # 提取前100个匹配的关键点（可根据需要调整）
    good_matches = matches[:100]

    ##绘制匹配结果
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

    ## 显示匹配图像

    img1_kps = []
    img2_kps = []

    for i in good_matches:
        key1_idx = i.queryIdx
        key2_idx = i.trainIdx

        pt1 = np.array(kp1[key1_idx].pt, dtype=np.int32)
        pt1[1] += place_cut_img[str(dir_id)]

        pt2 = np.array(kp2[key2_idx].pt, dtype=np.int32)
        pt2[1] += place_cut_img[str(dir_id)]
        img1_kps.append(pt1)
        img2_kps.append(pt2)
        # print(key1_idx, key2_idx)
        # print(kp1[key1_idx].pt, kp2[key2_idx].pt)

        # cv2.circle(img1_c, pt1, 1, color=[0,  20, 60], thickness=2)
        # cv2.circle(img2_c, pt2, 1, color=[20, 60, 0], thickness=2)

    #     cv2.imshow("img1", img1_c)
    #     cv2.imshow("img2", img2_c)
    #     cv2.imshow("Matched Keypoints with SIFT", img3)
    #     cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    return np.array(img1_kps,dtype=np.float32), np.array(img2_kps,dtype=np.float32)


def calculate_fundamental_matrix(keypoints1, keypoints2):
    """
    计算两幅图像间的基础矩阵。

    参数:
    - img1, img2: 两幅输入图像，作为numpy数组或cv2读取的图像。
    - keypoints1, keypoints2: 在两幅图像上对应的特征点列表，每个点应为(x, y)格式。

    返回:
    - F: 基础矩阵，如果计算成功；否则返回None。
    """
    # 确保特征点数量足够
    if len(keypoints1) < 8 or len(keypoints2) < 8:
        print("特征点数量不足，至少需要8对点来计算基础矩阵。")
        return None

    # 将关键点转换为OpenCV需要的格式
    points1 = np.float32(keypoints1).reshape(-1, 1, 2)
    points2 = np.float32(keypoints2).reshape(-1, 1, 2)

    # 使用OpenCV的findFundamentalMat函数计算基础矩阵，这里采用RANSAC方法来提高鲁棒性
    method = cv2.FM_RANSAC  # 可以选择其他方法，如FM_LMEDS等
    F, inliers = cv2.findFundamentalMat(points1, points2, method=method, ransacReprojThreshold=3.0)

    # 检查基础矩阵是否成功计算
    # if F is not None:
    #     print(f"成功找到基础矩阵，内点比例: {sum(inliers) / len(inliers)}")
    # else:
    #     print("未能成功计算基础矩阵。")

    return F


def estimate_camera_pose(F, K, points1, points2):
    # 从基础矩阵F和内参K计算本质矩阵E
    E = K.T @ F @ K
    # 确保E是秩为3的矩阵，有时候需要对E进行奇异值分解并重新构建以确保这一点
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]
    E = U @ np.diag(S) @ Vt

    # 使用RANSAC来恢复旋转和平移
    _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=K)
    return R, t

def estimate_player_pose(K, R, t, points):
    """
    :param K:  3X3
    :param R: 3X3
    :param t: 1X3
    :param points: N * 3
    :return:
    """

    points = np.concatenate((points, np.ones(shape=[len(points), 1])), axis=1)
    is_zero = np.sum(points,axis=1) == 1
    #print(points)
    K_inv = np.linalg.inv(K)
    points_nor = np.dot(K_inv, points.T).T
    R_inv = np.linalg.inv(R)
    pose_3D = np.dot(R_inv, points_nor.T).T - np.dot(R_inv, t.T).T
    pose_3D[is_zero] = 0
    return pose_3D

# x, y = -0.2856, -0.7353
# R = np.array([[0.19336179, 0.51387006, -0.8357923],
#               [0.5183005, -0.77680203, -0.35769149],
#               [-0.83305211, -0.3640277, -0.41654294]])
# t = np.array([0.77330021, 0.33392298, -0.53898259])

# pose_3D = estimate_player_pose(np.array([[734.8000053 ,   0.        , 360.        ],[  0.        , 734.21620003, 640.        ],[0,0,1]]),
#                                np.array([[ 0.19336179,  0.51387006, -0.8357923 ],[ 0.5183005,  -0.77680203, -0.35769149], [-0.83305211, -0.3640277,  -0.41654294]]),
#                                np.array([[ 0.77330021],[ 0.33392298],[-0.53898259]]),
#                                np.array([[150.0,200.0],[50.0,500.0]])
#                                )
#
# print(pose_3D)

# Ks = get_intriMetrix()
# print(Ks)

if __name__ == "__main__":
    place_cut_img = {"0": 250, "1": 300, "2": 420, "3": 370, "4": 370, "5": 290, "6": 320, "7": 320, "8": 280, "9": 300,
                     "10": 280, "11": 310, "12": 320, "13": 380,
                     "14": 300, "15": 310, "16": 300, "17": 340, "18": 270, "19": 310, "20": 350, "21": 320, "22": 330,
                     "23": 350, "24": 300, "25": 260, "26": 340,
                     "27": 320, "28": 260, "29": 200, "30": 300, "31": 260, "32": 240, "33": 360, "34": 300, "35": 280,
                     "36": 280, "37": 370, "38": 280, "39": 280,
                     "40": 410, "41": 430, "42": 280, "43": 410, "44": 480, "45": 460, "46": 190, "47": 230, "48": 250,
                     "49": 230, "50": 180, "51": 250, "52": 260, "53": 330, "54": 34}

    Ks = get_intriMetrix()  #获取每组图像的内参矩阵
    dir_src = r"E:\Learning\MyPaperData\PaperOne\videos_completor_pose"
    dir_base_location = r"E:\Learning\MyPaperData\PaperOne\videos_completor_places"
    dir_dst = r"E:\Learning\MyPaperData\PaperOne\videos_competor_pose_3d"
    dir_full_pic = r"E:\Learning\MyPaperData\PaperOne\videos"


    for dir in os.listdir(dir_base_location)[29:]:
        try:
            os.mkdir(dir_dst + "/" + dir)
        except:
            pass
        for subdir in os.listdir(dir_base_location+"/"+dir):
            count = 0
            last_pic = None
            try:
                os.mkdir(dir_dst + "/" + dir+ "/" + subdir)
            except:
                pass
            a = time.time()
            for json_file in os.listdir(dir_base_location+"/"+dir + "/"+subdir):
                place_pose_path = dir_base_location+"/"+dir + "/"+subdir + "/"  + json_file
                full_pic_path = dir_full_pic +"/"+ dir + "/"+subdir + "/" + json_file.split(".")[0] + ".jpg"
                src_pose_path = dir_src + "/" + dir + "/" + subdir + "/" + json_file
                dst_pose_path = dir_dst + "/" + dir + "/" + subdir + "/" + json_file
                base_json = np.array(json.loads(json.load(open(place_pose_path,'r'))),dtype=np.int32)
                try:
                    src_pose_json = json.loads(json.load(open(src_pose_path,'r')))
                    base_json[:, 1] = base_json[:, 1] + place_cut_img[dir]
                    base_json[:, 3] = base_json[:, 3] + place_cut_img[dir]
                    # print(base_json)
                    # print(full_pic_path)
                    # print(src_pose_json)
                    # print()

                    R_org = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
                    t_org = np.array([[0, 0, 0, 1]], dtype=np.float32)
                    R_t_matrix = np.concatenate((R_org, t_org.reshape([-1, 1])), axis=1)
                    img = cv2.imread(full_pic_path, 1)

                    poses_4_json = {}
                    if count == 0:
                        for i in base_json:
                            try:
                                temp_src_pose = np.array(src_pose_json[str(i[-1])])[:, :-1]
                                temp_src_pose[:, 0][temp_src_pose[:, 0] != 0] = temp_src_pose[:, 0][temp_src_pose[:, 0] != 0] + \
                                                                                i[0]
                                temp_src_pose[:, 1][temp_src_pose[:, 1] != 0] = temp_src_pose[:, 1][temp_src_pose[:, 1] != 0] + \
                                                                                i[1]
                                # cv2.rectangle(img,i[0:2],i[2:4],(50,0,200),1)
                            #     for j in temp_src_pose:
                            #         cv2.circle(img, np.array(j, dtype=np.int32), 1,(50, 0, 200), 2)
                            # cv2.imshow("img",img)
                            # cv2.waitKey(1000)
                                pose = estimate_player_pose(Ks[dir], R_org[:3], t_org[:,:3], temp_src_pose)
                                poses_4_json[str(i[-1])] = pose.tolist()
                            except:
                                pass

                    else:
                        kp1, kp2 = match_keypoints_sift(last_pic, img, place_cut_img,0)

                        if len(kp1) == 0:
                            R_t_matrix = np.concatenate((R_org, t_org.reshape([-1,1])),axis=1)

                        else:

                            fm = calculate_fundamental_matrix(kp1, kp2)  #基础矩阵的秩为2
                            R,t = estimate_camera_pose(fm, Ks[dir], kp1, kp2) #获取图像2相对于图像1的相机位姿
                            R_t_matrix_temp = np.concatenate((np.concatenate((R,[[0,0,0]]),axis=0),
                                                              np.concatenate((t,[[1]]),axis=0)),axis=1)

                            R_t_matrix = np.dot(R_t_matrix, R_t_matrix_temp)

                        R_new = R_t_matrix[0:3,0:3]
                        t_new = R_t_matrix[0:3,3].reshape([1,3])
                        for i in base_json:
                            try:
                                temp_src_pose = np.array(src_pose_json[str(i[-1])])[:, :-1]
                                temp_src_pose[:, 0][temp_src_pose[:, 0] != 0] = temp_src_pose[:, 0][temp_src_pose[:, 0] != 0] + \
                                                                                i[0]
                                temp_src_pose[:, 1][temp_src_pose[:, 1] != 0] = temp_src_pose[:, 1][temp_src_pose[:, 1] != 0] + \
                                                                                i[1]
                                pose = estimate_player_pose(Ks[dir], R_new, t_new, temp_src_pose)
                                poses_4_json[str(i[-1])] = pose.tolist()
                            except:
                                pass
                        # print(R_t_matrix)
                        # print(R_new)
                        # print(t_new)
                    count+=1
                    last_pic = img.copy()
                    f = open(dst_pose_path, "w")
                    f.write(json.dumps(poses_4_json))
                    f.close()
                except:
                    pass
            b = time.time()
            print(dir, ":\t",subdir)
            print("time_use: ", b-a)




