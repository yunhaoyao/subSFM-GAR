import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np


class MyTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, feedforward_dim, dropout=0.1):
        super(MyTransformerEncoder, self).__init__()

        self.projection_layer = nn.Linear(input_dim, output_dim)
        self.active = nn.LeakyReLU(0.1)
        encoder_layers = TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
                                                 dropout=dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):

        y = self.active(self.projection_layer(src))
        output = self.transformer_encoder(y)

        return output

class MySwinTransformerEncoder(nn.Module):
    def __init__(self,device, img_size, out_dim):
        super(MySwinTransformerEncoder, self).__init__() #input pic size = [N,C,H,W]=[N,3,384,384]
        self.patch_sizes = [8,12,16,24,32]
        self.img_size = img_size
        self.seqence_length = [int((self.img_size / x) ** 2) for x in self.patch_sizes]
        self.input_dims = [x ** 2 * 3 for x in self.patch_sizes]
        self.out_dim = out_dim
        self.models = [MyTransformerEncoder(input_dim, self.out_dim, 8, 5, 2048, 0.1) for input_dim in self.input_dims]
        self.device = device
        for model in self.models:
            model.to(self.device)

    def forward(self,x):
        ##x.shape = [N,3,384,384]
        is_1st = True
        for i in range(len(self.patch_sizes)):
            x_temp = x.reshape([self.seqence_length[i], -1, self.patch_sizes[i] * self.patch_sizes[i] * 3])
            if is_1st:
                out = self.models[i].forward(x_temp)
                is_1st = False
            else:
                out_temp = self.models[i].forward(x_temp)
                out = torch.concatenate((out,out_temp),dim=0)
        return out.permute([1,0,2])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# ##假设输入数据形状为 (sequence_length=10, batch_size=32, input_dim=512)
# batch_size = 14
# img_size = 192
# out_dim = 1024
# swin = MySwinTransformerEncoder(device,img_size,out_dim)
# for i in range(100):
#     data = np.array(np.random.randint(0,255,batch_size*img_size*img_size*3),dtype=np.float32).reshape([batch_size,img_size,img_size,3]) / 255.
#     data = torch.tensor(data,dtype=torch.float32).to(device)
#     output = swin.forward(data)
#     print(output.shape)













