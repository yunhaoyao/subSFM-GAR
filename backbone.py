
import torch
import torch.nn as nn
from pytorch_ResUnit import TorchResUnit as res
import numpy as np
import transformer_test


class SrcPicRoute(nn.Module):
    def __init__(self, device):
        super(SrcPicRoute, self).__init__()
        self.device = device

        self.res1 = res(1024,     1,   8,  True, device = self.device )
        self.res2 = res(512,     8,   16,  True, device = self.device )
        self.res3 = res(256,     16,   32,  True, device = self.device )
        self.res4 = res(128,     32,   64,  True, device = self.device )
        self.res5 = res(64,     64,   128,  True, device = self.device )
        self.res6 = res(32,     128,   256,  True, device = self.device )
        self.res7 = res(16, 256, 512, True, device=self.device)
        self.res8 = res(8, 512, 512, True, device=self.device)
        self.res9_1 = res(4, 512, 512, True, device=self.device)
        self.res10_1 = res(2, 512, 256, True, device=self.device)
        self.conv_9_2 = torch.nn.Conv2d(512, 512, 3,
                                         stride=2,
                                         padding=1,
                                         dilation=1,
                                         groups=1,
                                         bias=True,
                                         padding_mode='zeros',
                                         device=self.device)
        self.conv_10_2 = torch.nn.Conv2d(512, 1, 3,
                                        stride=1,
                                        padding=1,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros',
                                        device=self.device)
        self.out_act = nn.Tanh()

    def forward(self,X):
        X = torch.reshape(X,[-1,1,1076,1024])
        y1 = self.res1.forward(X)
        y2 = self.res2.forward(y1)
        y3 = self.res3.forward(y2)
        y4 = self.res4.forward(y3)
        y5 = self.res5.forward(y4)
        y6 = self.res6.forward(y5)
        y7 = self.res7.forward(y6) #
        y8 = self.res8.forward(y7)
        y8_pad = torch.concatenate((y8,torch.zeros(size=[X.shape[0],512,5,1]).to(self.device)),dim=-1)

        y9_1 = self.res9_1.forward(y8)
        y9_2 = self.conv_9_2(y8_pad)

        y10_1 = self.res10_1.forward(y9_1)
        y10_2 = self.conv_10_2(y9_2).permute([1, 0, 2, 3])

        return torch.reshape(y10_1,[X.shape[0],-1]), y10_2

class PosePointCloudRoute(nn.Module):
    def __init__(self, device):
        super(PosePointCloudRoute, self).__init__()  # 首先调用父类构造函数
        self.device = device  # 然后设置 device
        self.res1 = res(4, 1, 8, True, device=self.device)
        self.res2 = res(4, 8, 16, True, device=self.device)
        self.res3 = res(2, 16, 32, True, device=self.device)
        self.res4 = res(2, 32, 32, True, device=self.device)

    def forward(self,X, rotations):
        ####X.shape = (n * 41 * 20, 17, 3)
        ##### rotations.shape = (n, 41, 3, 3) ## n = batch_size
        X = torch.reshape(X,shape=[-1,41,340,3])
        #X = torch.einsum('ijlm, ijmn -> ijln', X, rotations)
        X = X.permute([1,0,2,3])
        y1 = torch.reshape(self.res1.forward(X), [41, 8, 85, 4])
        y2 = self.res2.forward(y1)
        y3 = torch.reshape(self.res3.forward(y2),[41, 32, 11, 2])
        y4 = self.res4.forward(y3)
        return torch.reshape(y4,[41,-1])


class SporterRoute(nn.Module):
    def __init__(self, device):
        super(SporterRoute, self).__init__()
        self.device = device
        self.res1 = res(96, 60, 128, True, device=self.device)
        self.res2 = res(48, 128, 256, True, device=self.device)
        self.res3 = res(24, 256, 512, False, device=self.device)
        self.res4 = res(24, 512, 512, False, device=self.device)
        self.res5 = res(24, 512, 256, True, device=self.device)
        self.res6 = res(12, 256, 128, True, device=self.device)
        self.res7 = res(6, 128, 32, True, device=self.device)

    def forward(self,X):
        #X.shape = (328, 20, 96, 96, 3)
        y1 = self.res1.forward(X)
        y2 = self.res2.forward(y1)
        y3 = self.res3.forward(y2)
        y4 = self.res4.forward(y3)
        y5 = self.res5.forward(y4)
        y6 = self.res6.forward(y5)
        y7 = self.res7.forward(y6)

        return torch.reshape(y7,[41,-1])

class LSTMModel(nn.Module):
    def __init__(self,device, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.active = nn.LeakyReLU(0.1)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.active(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out


class BackBone(nn.Module):
    def __init__(self, device, batch_size):
        super(BackBone, self).__init__()  # 首先调用父类构造函数
        self.device = device
        self.batch_size = batch_size
        self.srcPicRoute = SrcPicRoute(self.device)
        self.sporterRoute = SporterRoute(self.device)
        self.posePointCloudRoute = PosePointCloudRoute(self.device)
        self.T_encoder = transformer_test.MySwinTransformerEncoder(self.device,192,1024)
        #self.LSTM_out = LSTMModel(self.device, 992, 1024, 3, 8).to(self.device)
        self.LSTM_out = LSTMModel(self.device, 512, 1024, 3, 8).to(self.device)
        self.sigmoid = nn.Softmax()

    def forward(self, srcPic_seq, sporter_pic_src, posePointcloud_feature, sub_batch_size):
        """
        :param srcPic_feature:
        :param sporter_feature:
        :param posePointcloud_feature:
        :return:
        """
        srcPic_feature = None
        N = 41//sub_batch_size
        for i in range(N): #每次7张图
            if i==0:
                srcPic_feature = self.T_encoder.forward(srcPic_seq[i*sub_batch_size:(i+1)*sub_batch_size])
            else:
                srcPic_feature = torch.concatenate((srcPic_feature,self.T_encoder.forward(srcPic_seq[i*sub_batch_size:(i+1)*sub_batch_size])),dim=0)

        srcPic_seq = None
        pic_feature = None
        rotations = None
        for i in range(N):  # 每次7张图
            if i == 0:
                pic_feature, rotations = self.srcPicRoute.forward(srcPic_feature[i*sub_batch_size:(i+1)*sub_batch_size])
                pic_feature = pic_feature.detach().cpu().numpy()
                rotations = rotations.detach().cpu().numpy()
            else:
                pic_feature_temp, rotations_temp = self.srcPicRoute.forward(srcPic_feature[i * sub_batch_size:(i + 1) * sub_batch_size])
                pic_feature_temp = pic_feature_temp.detach().cpu().numpy()
                rotations_temp = rotations_temp.detach().cpu().numpy()

                pic_feature = np.concatenate((pic_feature, pic_feature_temp),axis=0)
                rotations = np.concatenate((rotations, rotations_temp), axis=1)

                # pic_feature = torch.concatenate((pic_feature,pic_feature_temp),dim=0)
                # rotations = torch.concatenate((rotations, rotations_temp), dim=0)
        srcPic_feature = None
        pic_feature = torch.tensor(pic_feature).to(self.device)
        # rotations = torch.tensor(rotations).to(self.device)
        # sporter_feature = self.sporterRoute.forward(sporter_pic_src)
        sporter_pic_src = None
        # pose_feature = self.posePointCloudRoute.forward(posePointcloud_feature, rotations)
        # data_comcated = torch.concatenate((pic_feature,sporter_feature,pose_feature),dim=-1)
        # out = self.sigmoid(self.LSTM_out.forward(torch.reshape(data_comcated,[-1,41,992])))

        out = self.sigmoid(self.LSTM_out.forward(torch.reshape(pic_feature, [-1, 41, 512])))
        return out, rotations


# # 检查是否有GPU可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# for i in range(10):
#     pic_src_seq = np.array(np.random.randint(0,255,41*192*192*3),dtype=np.float32).reshape([41,192,192,3]) / 255.
#     pic_src_seq = torch.tensor(pic_src_seq,dtype=torch.float32).to(device)
#     posePointCloud_src = torch.tensor(np.random.normal(size=[41*20, 17, 3]),dtype=torch.float32).to(device)
#     sporter_pic_src = torch.tensor(np.random.normal(size=[41, 60, 96, 96]),dtype=torch.float32).to(device)
#
#     model = BackBone(device,1)
#     model.forward(pic_src_seq, sporter_pic_src, posePointCloud_src,1)



#这是流量包静态推荐模型的训练代码
