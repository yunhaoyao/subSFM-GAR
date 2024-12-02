import time

from get_train_files import *
from backbone import BackBone
import numpy as np
import cv2
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CustomMSELoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomMSELoss, self).__init__()
        # 这里可以选择不同的 reduction 方式
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, rotation_loss):
        # 计算预测值和真实值之间的差值的平方
        loss = self.criterion(y_pred, y_true)

        # 根据 reduction 参数决定如何聚合损失
        if self.reduction == 'mean':
            return loss.mean()  #* 0 + (rotation_loss / 41.0)
        elif self.reduction == 'sum':
            return loss.sum() #* 0 + (rotation_loss / 41.0)
        else:
            return loss.sum() #* 0 + (rotation_loss / 41.0)


# 创建自定义损失函数实例

class Train:
    def __init__(self,device,epoch):
        super(Train).__init__()
        self.device = device
        self.epoch = epoch
        self.batch_size = 1
        self.model = BackBone(device, self.batch_size)
        self.custom_loss = CustomMSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)
        self.ax = []
        self.ay = []

    def train(self):
        try:
            self.model.load_state_dict(torch.load('model_params.pth'))
            self.model.train()
            print("successfully loaded model")
        except:
            print("new model!")
        #{3: 10, 4: 2, 2: 25, 0: 19, 5: 32, 6: 27, 1: 18, 7: 20}
        train_dirs = r"1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54".split(" ")[:1]
        validation_dirs = r"0 2 8 12 17 19 24 26 27 28 30 33 46 49 51".split(" ")
        test_dirs = r"4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47".split(" ")

        video_dirs = r"E:\Learning\MyPaperData\PaperOne\videos"
        competitor_dirs = r"E:\Learning\MyPaperData\PaperOne\videos_completor_only"
        pose_dirs = r"E:\Learning\MyPaperData\PaperOne\videos_competor_pose_3d"
        label_path = r"E:\Learning\MyPaperData\PaperOne\labels"

        # labels = {"r_set":0, "r_spike":1,  "r_pass":2, "r_winpoint":3, "l_winpoint":4, "l_pass":5, "l_spike":6, "l_set":7,
        #     "r-set":0, "r-spike":1,  "r-pass":2, "r-winpoint":3, "l-winpoint":4, "l-pass":5, "l-spike":6, "l-set":7}
        train_count = 0
        count = 0
        batch_size = self.batch_size
        src_pic_seq = []
        person_pic_seq = []
        pos_ord_seq = []
        labels = []
        epoch = self.epoch
        a = time.time()
        for ep in range(epoch):
            np.random.shuffle(train_dirs)
            for dir in train_dirs:
                #print("dir: ", dir)
                label_file = json.load(open(label_path + "/" + dir + ".json", "r"))
                # print(label_file)
                video_dir = video_dirs + "/" + dir
                competitor_dir = competitor_dirs + "/" + dir
                pose_dir = pose_dirs + "/" + dir
                files = list(os.walk(video_dir))[0][1]
                np.random.shuffle(files)
                for file_dir in files:
                    # if label_file[file_dir] == 5 and np.random.randint(0, 10) < 3:
                    #     continue
                    labels.append(label_file[file_dir])
                    competitor_dir_sub = competitor_dir + "/" + file_dir
                    pose_dir_sub = pose_dir + "/" + file_dir
                    # print("file_dir: ", file_dir)
                    file_name = list(os.walk(video_dir + "/" + file_dir))[0][2]
                    #print(file_name)
                    file_name_digital = [str(y) + ".jpg" for y in sorted([int(x.split(".")[0]) for x in file_name])]
                    for pic_file_name in file_name_digital:
                        img_temp = resize_pic(cv2.imread(video_dir + "/" + file_dir + "/" + pic_file_name, 1), 192)
                        try:
                            pose_dect_file = json.load(open(pose_dir_sub + "/" + pic_file_name[:-4] + ".json", "r"))
                        except:
                            pose_dect_file = None
                        # print(pose_dect_file)
                        if img_temp.shape != (192, 192, 3):
                            print(img_temp.shape)
                            print("ssssssssss")
                            quit()
                        src_pic_seq.append(img_temp)
                        competitor_dir_path = competitor_dir_sub + "/" + pic_file_name[:-4]
                        # print("competitor_dir_path: ",competitor_dir_path)
                        competitors_file = list(os.walk(competitor_dir_path))[0][2]
                        competitors_pic_temp = []
                        for c_f in competitors_file[:20]:
                            competitor_file_path = competitor_dir_path + "/" + c_f
                            try:
                                img_temp2 = resize_pic(cv2.imread(competitor_file_path, 1), 96)
                                cp_pose = pose_dect_file[c_f[:-4]][:20]
                            except:
                                img_temp2 = np.zeros(shape=[96, 96, 3], dtype=np.float32)
                                cp_pose = np.zeros(shape=[17, 3], dtype=np.float32)
                            competitors_pic_temp.append(img_temp2)
                            pos_ord_seq.append(cp_pose)


                        if len(competitors_pic_temp) < 20:
                            for n in range(20 - len(competitors_pic_temp)):
                                competitors_pic_temp.append(np.zeros([96, 96, 3], dtype=np.float32))
                                pos_ord_seq.append(np.zeros(shape=[17, 3], dtype=np.float32))
                        person_pic_seq.append(competitors_pic_temp)
                    count += 1
                    if count == batch_size:
                        labels = trans_label_2_onehot(labels)

                        srcPic_seq = torch.tensor(np.array(src_pic_seq,dtype=np.float32).reshape([41, 192, 192, 3]) / 255,dtype=torch.float32).to(self.device)
                        posePointCloud_src = torch.tensor(np.array(pos_ord_seq,dtype=np.float32),dtype=torch.float32).to(self.device) / 3
                        print(torch.max(posePointCloud_src))
                        sporter_pic_src = torch.tensor(np.array(person_pic_seq,dtype=np.float32) / 255, dtype=torch.float32).to(self.device)
                        sporter_pic_src = sporter_pic_src.permute([0,2,3,4,1])
                        sporter_pic_src = torch.reshape(sporter_pic_src,[-1,96,96,60])
                        sporter_pic_src = sporter_pic_src.permute([0,3,1,2])

                        class_out, rotations_out = self.model.forward(srcPic_seq, sporter_pic_src, posePointCloud_src, 1)

                        rotation_loss = 0
                        determinants = []
                        rotations = []
                        # for k in range(41):
                        #     determinant =(torch.abs(torch.linalg.det(rotations_out[0][k])) - 1) ** 2
                        #     rotations.append(rotations_out[0][k])
                        #     determinants.append(determinant)
                        #     rotation_loss += determinant

                        loss = self.custom_loss.forward(class_out,torch.tensor(labels,dtype=torch.float32).to(self.device),rotation_loss)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        print("pred: ", class_out)
                        print("label: ", labels)

                        if train_count % 10 == 0:
                            self.ax.append(train_count)
                            self.ay.append(loss.detach().cpu().numpy())
                            b = time.time()
                            print(train_count, ": ", loss, "time: ", b - a, "Roloss: ", rotation_loss)
                            print("pred: ", class_out)
                            print("label: ", labels)
                            # print(determinants)
                            # print(rotations[:2])
                            a = b
                            if train_count % 500 == 0:
                                torch.save(self.model.state_dict(), 'model_params.pth')
                            # plt.plot(self.ax,self.ay)
                            # plt.pause(0.1)
                        count = 0
                        src_pic_seq = []
                        labels = []
                        person_pic_seq = []
                        pos_ord_seq = []
                        train_count += 1

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
trainer = Train(device,500)
trainer.train()










