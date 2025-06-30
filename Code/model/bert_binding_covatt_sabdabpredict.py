# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoModel,RoFormerModel
import math
from ban16 import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import numpy as np


class BERTBinding_AbDab_cnn(nn.Module):
    def __init__(self, heavy_dir, light_dir, antigen_dir, emb_dim=256,  in_channel=118,ab_emb_dim=820, ag_emb_dim=692, h_dim=512, n_heads=2, output_dim=2, dropout=0.2,  attention=False):
        super().__init__()
        self.HeavyModel = AutoModel.from_pretrained(heavy_dir, output_hidden_states=True, return_dict=True)
        self.LightModel = AutoModel.from_pretrained(light_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,
                                                      cache_dir="../cache")
        self.multi_ppimi = MultiPPIMI(ab_emb_dim, ag_emb_dim, h_dim, n_heads, output_dim, dropout, attention)
        self.fc1_ag = nn.Linear(32 * 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.conv1_ag = MF_CNN_ag(692, 64)
        self.ConvGlobalAvgPoolingNet = ConvGlobalAvgPoolingNet()
        self.conv2 = MF_CNN_ag(64, 32)
        self.conv3 = MF_CNN_ag(32, 32)
        self.cnn1 = MF_CNN(in_channel=140)
        self.cnn2 = MF_CNN(in_channel=140)
        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128*4, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, heavy, light, antigen, H_dict, L_dict, antigen_dict, i):
        heavy_encoded = self.HeavyModel(**heavy).last_hidden_state
        # # 将 Tensor 转换为 numpy 数组
        # heavy_encoded_np = torch.tensor(heavy_encoded).squeeze(0)
        # heavy_encoded_np = heavy_encoded_np.detach().cpu().numpy()
        # # 定义保存路径
        # save_dir = "/share/home/zhanglab/lzx/deepaai/Code/save_feature/heavy"
        # os.makedirs(save_dir, exist_ok=True) 
        # save_path = os.path.join(save_dir, f"heavy_feature_{i}.csv")
        # df = pd.DataFrame(heavy_encoded_np)  # 
        # df.to_csv(save_path, index=False, header=False)  # 不保存索引和列名
        first_value = list(H_dict.values())[i]
        # data_tensor_H = torch.tensor(first_value)
        first_value = np.array(first_value)  # 将列表转换为 NumPy 数组
        data_tensor_H = torch.tensor(first_value).unsqueeze(0) # 再从 NumPy 数组创建 PyTorch 张量
        reshaped_tensor_expanded = torch.cat((torch.zeros((data_tensor_H.shape[0], 1, data_tensor_H.shape[2]), device=data_tensor_H.device), data_tensor_H), dim=1)
        reshaped_tensor_final = reshaped_tensor_expanded[:, :-1, :]
        reshaped_tensor = reshaped_tensor_final.to('cuda:0')
        # 将 Tensor 转换为 numpy 数组
        # reshaped_tensor_np = torch.tensor(reshaped_tensor).squeeze(0)
        # reshaped_tensor_np = reshaped_tensor_np.detach().cpu().numpy()
        # 定义保存路径
        # save_dir = "/share/home/zhanglab/lzx/deepaai/Code/save_feature/heavy_ph"
        # os.makedirs(save_dir, exist_ok=True) 
        # save_path = os.path.join(save_dir, f"ph_h_tensor_{i}.csv")
        # df = pd.DataFrame(reshaped_tensor_np)  # 
        # df.to_csv(save_path, index=False, header=False)  # 不保存索引和列名
        heavy_encoded = torch.cat((reshaped_tensor, heavy_encoded), dim=2)
        # print("heavy1",heavy_encoded.shape)

        light_encoded = self.LightModel(**light).last_hidden_state
        # light_encoded_np = torch.tensor(light_encoded).squeeze(0)
        # light_encoded_np = light_encoded_np.detach().cpu().numpy()
        # # 定义保存路径
        # save_dir = "/share/home/zhanglab/lzx/deepaai/Code/save_feature/light"
        # os.makedirs(save_dir, exist_ok=True) 
        # save_path = os.path.join(save_dir, f"light_feature_{i}.csv")
        # df = pd.DataFrame(light_encoded_np)  # 
        # df.to_csv(save_path, index=False, header=False)  
        first_value = list(L_dict.values())[i]
        # data_tensor_L = torch.tensor(first_value)
        first_value = np.array(first_value)  # 将列表转换为 NumPy 数组
        data_tensor_L = torch.tensor(first_value).unsqueeze(0)
        reshaped_tensor_expanded = torch.cat((torch.zeros((data_tensor_L.shape[0], 1, data_tensor_L.shape[2]), device=data_tensor_L.device), data_tensor_L), dim=1)
        reshaped_tensor_final = reshaped_tensor_expanded[:, :-1, :]
        reshaped_tensor = reshaped_tensor_final.to('cuda:0')
        light_encoded = torch.cat((reshaped_tensor, light_encoded), dim=2)
        # print("light1", light_encoded.shape)

        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state
        first_value = list(antigen_dict.values())[i]
        data_tensor_Ag = torch.tensor(first_value).unsqueeze(0)
        # data_tensor_ag = torch.tensor(data_tensor_Ag)
        data_tensor_ag = data_tensor_Ag.clone().detach()
        # print("antigen_encoded0", reshaped_tensor.shape)
        reshaped_tensor_expanded = torch.cat((torch.zeros((data_tensor_ag.shape[0], 1, data_tensor_ag.shape[2]), device=data_tensor_ag.device), data_tensor_ag), dim=1)
        reshaped_tensor_final = reshaped_tensor_expanded[:, :-1, :]
        # reshaped_tensor_final[:, -1, :] = 0
        reshaped_tensor = reshaped_tensor_final.to('cuda:0')
        # 将 Tensor 转换为 numpy 数组
        # reshaped_tensor_np = torch.tensor(reshaped_tensor).squeeze(0)
        # reshaped_tensor_np = reshaped_tensor_np.detach().cpu().numpy()
        # 定义保存路径
        # save_dir = "/share/home/zhanglab/lzx/deepaai/Code/save_feature/antigen_ph"
        # os.makedirs(save_dir, exist_ok=True) 
        # save_path = os.path.join(save_dir, f"ph_ag_tensor_{i}.csv")
        # df = pd.DataFrame(reshaped_tensor_np)  # 
        # df.to_csv(save_path, index=False, header=False)  # 不保存索引和列名
        # reshaped_tensor = torch.cat((reshaped_tensor, torch.zeros_like(reshaped_tensor[:, :2, :])), dim=1)
        L = reshaped_tensor.shape[1]
        antigen_encoded = antigen_encoded[:, :L, :]
        # antigen_encoded_np = torch.tensor(antigen_encoded).squeeze(0)
        # antigen_encoded_np = antigen_encoded_np.detach().cpu().numpy()
        # 定义保存路径
        # save_dir = "/share/home/zhanglab/lzx/deepaai/Code/save_feature/antigen"
        # os.makedirs(save_dir, exist_ok=True) 
        # save_path = os.path.join(save_dir, f"antigen_feature_{i}.csv")
        # df = pd.DataFrame(antigen_encoded_np)  # 
        # df.to_csv(save_path, index=False, header=False) 
        antigen_encoded0 = torch.cat((reshaped_tensor, antigen_encoded), dim=2)
        # print("antigen_encoded", antigen_encoded.shape)
        antigen_encoded = antigen_encoded0.permute(0, 2, 1)
        # batch_size, in_channel, seq_len = antigen_encoded.size()
        # print("antigen1", antigen_encoded.shape)
        antigen_encoded = self.conv1_ag(antigen_encoded)
        # print("antigen_encoded1", antigen_encoded.shape)
        antigen_encoded = self.conv2(antigen_encoded)
        # print("antigen_encoded2", antigen_encoded.shape)
        antigen_encoded = self.conv3(antigen_encoded)
        # print("antigen_encoded3", antigen_encoded.shape)
        antigen_encoded = antigen_encoded.mean(dim=2)
        antigen_encoded = nn.ReLU()(self.fc1_ag(antigen_encoded))
        sk = antigen_encoded
        antigen_encoded = self.fc2(antigen_encoded)
        antigen_encoded = self.fc3(antigen_encoded)
        antigen_cls = antigen_encoded + sk
        # print("antigen_encoded", antigen_encoded.shape)
        ab_att = torch.cat((heavy_encoded, light_encoded), 1)
        # print("ab_att", ab_att.shape)
        ab_att = ab_att.float()
        # print("ab_att", ab_att.shape)
        ag_att = antigen_encoded0
        ag_att = ag_att.float()
        # print("ag_attdtype:", ag_att.dtype)
        # print("ag_att", ag_att.shape)
        abag_att_cls = self.multi_ppimi(ab_att, ag_att)
        # print("abag_att_cls", abag_att_cls.shape)
        heavy_cls = self.cnn1(heavy_encoded)
        # print("heavy_cls", heavy_cls.shape)
        light_cls = self.cnn2(light_encoded)
        # print("light_cls", light_cls.shape)
        # antigen_cls = self.cnn3(antigen_encoded)
        # print("antigen_cls", antigen_cls.shape)
# feature——————————phcm
        # abag_phc = abag_dict[i:i+8]
        # abag_phc = pad_tensors_to_max_abag(abag_phc)
        # abag_phc = np.transpose(abag_phc, (0, 3, 1, 2))
        # abag_phc = torch.tensor(abag_phc).to('cuda:0')
        # print("abag_phc",abag_phc.shape)
        # abag_phc_cls = self.ConvGlobalAvgPoolingNet(abag_phc)
        # print("abag_phc_clsd", abag_phc_cls.shape)
        concated_encoded = torch.concat((heavy_cls, light_cls, antigen_cls, abag_att_cls), dim=1)
        # concated_encoded = torch.concat((heavy_cls, light_cls, antigen_cls, abag_att_cls,abag_phc_cls), dim=1)
        # concated_encoded = torch.concat((heavy_cls, light_cls, antigen_cls), dim=1)
        # print("concated_encoded", concated_encoded.shape)

        output = self.binding_predict(concated_encoded)

        return output

class MultiPPIMI(nn.Module):
    def __init__(self, ab_emb_dim, ag_emb_dim,
                 h_dim, n_heads,
                 output_dim=2, dropout=0.2, attention=False):
        super(MultiPPIMI, self).__init__()
        self.attention = attention
        self.ab_emb_dim = ab_emb_dim
        self.ag_emb_dim = ag_emb_dim

        ##### bilinear attention #####
        self.bcn = weight_norm(
            BANLayer(v_dim=ab_emb_dim, q_dim=ag_emb_dim, h_dim=h_dim, h_out=n_heads, k=3),
            name='h_mat', dim=None)

        self.fc1 = nn.Linear(h_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, ab, ag):

        x, att = self.bcn(ab, ag)
        # print("x0", x.shape)
        x = self.fc1(x)
        # print("x1", x.shape)
        x = self.relu(x)
        # print("x2", x.shape)
        x = self.dropout(x)
        # print("x3", x.shape)
        x = self.fc2(x)
        # print("x4", x.shape)
        x = self.relu(x)
        # print("x5", x.shape)
        x = self.dropout(x)
        # print("x6", x.shape)
        x = self.out(x)
        return x

class ConvGlobalAvgPoolingNet(nn.Module):
    def __init__(self, input_channels=5, output_size=128):
        super(ConvGlobalAvgPoolingNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_max_pool(x)
        # print("x7", x.shape)
        x = x.view(x.size(0), -1)
        # print("x8", x.shape)
        x = self.fc(x)
        return x


class MF_CNN_ag(nn.Module):
    def __init__(self, in_channel, hidden_channel=2):
        super(MF_CNN_ag, self).__init__()

        self.cnn = nn.Conv1d(in_channel, hidden_channel, kernel_size=5, stride=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x


# 抗体侧cnn
class MF_CNN(nn.Module):
    def __init__(self, in_channel=118, emb_size=20, hidden_size=99):  # 189):
        super(MF_CNN, self).__init__()

        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn_liu(in_channel=in_channel, hidden_channel=64)  # 118*64
        self.conv2 = cnn_liu(in_channel=64, hidden_channel=32)  # 64*32

        self.conv3 = cnn_liu(in_channel=32, hidden_channel=32)

        self.fc1 = nn.Linear(32 * hidden_size, 128)  # 32*29*512
        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 128)

    def forward(self, x):
        # x = x
        # x = self.emb(x)

        x = self.conv1(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # (1, 384)
        # print(x.shape)
        x = nn.ReLU()(self.fc1(x))
        # print(x.shape)
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x + sk


class cnn_liu(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn_liu, self).__init__()

        self.cnn = nn.Conv1d(in_channel, hidden_channel, kernel_size=5, stride=1)  # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # bs * 32*30

        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.emb(x)
        x = x.float()
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x



def pad_tensors_to_max_length(tensor_list, max_length=800):
    shapes = [tensor.shape for tensor in tensor_list]
    max_seq_length = min(max(shape[0] for shape in shapes), max_length)
    padded_tensors = []
    for tensor in tensor_list:
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        current_length, feature_size = tensor.shape
        # batch_size, current_length, feature_size = tensor.shape
        if current_length < max_seq_length:
            padding_size = max_seq_length - current_length
            # 创建一个形状为 (batch_size, padding_size, feature_size) 的零张量
            padding_tensor = torch.zeros( padding_size, feature_size)
            # 将当前张量与零填充张量拼接
            padded_tensor = torch.cat([tensor, padding_tensor], dim=0)
        else:
            # 如果张量的第二维已经是最大值，则不需要填充
            padded_tensor = tensor[:max_seq_length, :]
        padded_tensors.append(padded_tensor)
    return  torch.stack(padded_tensors)



def pad_tensors_to_max_abag(tensors, max_L1=800):
    max_L1_in_batch = min(max_L1, max([tensor.shape[0] for tensor in tensors]))  # 限制 L1 最大值
    max_L2 = max([tensor.shape[1] for tensor in tensors])
    padded_tensors = []
    for tensor in tensors:
        # 创建一个形状为 (max_L1_in_batch, max_L2, 5) 的零张量
        padded_tensor = torch.zeros(max_L1_in_batch, max_L2, tensor.shape[2])
        # 计算需要截断的长度
        truncated_L1 = min(tensor.shape[0], max_L1_in_batch)
        truncated_L2 = min(tensor.shape[1], max_L2)
        # 将原始张量的部分（截断后的）复制到零张量的对应部分
        padded_tensor[:truncated_L1, :truncated_L2, :] = tensor[:truncated_L1, :truncated_L2, :]
        padded_tensors.append(padded_tensor)
    return padded_tensors