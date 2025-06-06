import torch
import torch.nn as nn
from moudel.timewise import PositionalEncoding
from moudel.multi_group_encoder import Multi_Group_encoders
from moudel.channelwise import ChannelTransformerModel

def truncate_data(x, mask):
    """
    根据 mask 截断数据，删除无效的时间步（即所有特征值为 0 的时间步）。
    :param x: 输入数据，形状为 (batch_size, seq_len, input_size)
    :param mask: 掩码，形状为 (batch_size, seq_len)，其中为 1 的位置表示有效的时间步
    :return: 截断后的数据，形状为 (batch_size, valid_seq_len, input_size)
    """
    # 使用掩码选择有效时间步
    valid_indices = mask.nonzero(as_tuple=False)[:, 1]  # 获取有效时间步的索引

    # 截断数据，仅保留有效的时间步
    truncated_data = [x[i, valid_indices[i]] for i in range(x.size(0))]

    # 将截断后的数据堆叠回一个新的张量
    truncated_data = torch.nn.utils.rnn.pad_sequence(truncated_data, batch_first=True)

    return truncated_data


class FeatureExtractor(nn.Module):
    """特征提取模块"""
    def __init__(self, input_size, output_size):
        super(FeatureExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.LeakyReLU()
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # 使用掩码屏蔽填充部分
        return self.extractor(x)



class MgMcFORMER(nn.Module):
    def __init__(self, multi_group, nclasses, seq_len, input_size, emb_size, nhid, emb_size_c, nhid_c, nhead, nlayers,
                 device='cpu', dropout=0.1):
        super(MgMcFORMER, self).__init__()
        # 特征提取模块
        feature_dim = input_size // 2  # 假设 g 和 r 波段的输入维度各占一半
        self.g_feature_extractor = FeatureExtractor(input_size=feature_dim, output_size=emb_size//2)
        self.r_feature_extractor = FeatureExtractor(input_size=feature_dim, output_size=emb_size//2)

        # trunk_net 现在接收拼接后的特征作为输入
        self.trunk_net = nn.Sequential(
            nn.Linear(emb_size , emb_size),  # 两个波段特征拼接后维度翻倍
            nn.LayerNorm(emb_size),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.LayerNorm(emb_size)
        )

        # Multi_group
        self.multi_group = Multi_Group_encoders(
            multi_group=multi_group, seq_len=seq_len, emb_size=emb_size, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout
        )

        # C_A
        self.c_a = ChannelTransformerModel(
            seq_len=seq_len, emb_size=emb_size_c, nhead=nhead, nhid=nhid_c, nlayers=nlayers, device=device, dropout=dropout
        )

        self.retrunk_net = nn.Sequential(
            nn.Linear(emb_size_c, seq_len),
            nn.LeakyReLU(),
            nn.LayerNorm(seq_len),
        )

        # class
        self.class_net = nn.Sequential(
            nn.Linear(seq_len * emb_size, nhid_c),
            nn.LeakyReLU(),
            nn.LayerNorm(nhid_c),
            nn.Dropout(p=0.3),
            nn.Linear(nhid_c, nhid_c),
            nn.LeakyReLU(),
            nn.LayerNorm(nhid_c),
            nn.Dropout(p=0.3),
            nn.Linear(nhid_c, nclasses)
        )

        self.laynorm = nn.LayerNorm(emb_size)

    def forward(self, x, mask=None):
        # 假设输入 x 的 shape 为 (batch_size, seq_len, input_size)
        # 将输入拆分为 g 波段和 r 波段
        half_dim = x.size(-1) // 2
        x_g, x_r = x[:, :, :half_dim], x[:, :, half_dim:]

        # 动态生成掩码：如果 g 或 r 波段的某个时间步的所有特征值为0，则该时间步被视为无效
        mask_g = (x_g.sum(dim=-1) != 0)  # g 波段掩码
        mask_r = (x_r.sum(dim=-1) != 0)  # r 波段掩码

        # 提取 g 和 r 波段的特征
        g_features = self.g_feature_extractor(x_g, mask_g)
        r_features = self.r_feature_extractor(x_r, mask_r)

        # 拼接 g 和 r 波段的特征
        combined_features = torch.cat([g_features, r_features], dim=-1)

        # 输入 trunk_net
        x = self.trunk_net(combined_features)

        # Multi_group
        out = self.multi_group(x)

        res = out
        if mask is not None:
            out = torch.where(mask, torch.as_tensor(0.0, device=x.device), out)

        # Channel-wise attention
        out, att = self.c_a(out)

        # retrunk_net
        out = self.retrunk_net(out).permute(0, 2, 1)
        out = self.laynorm(out + res)

        # 分类网络
        out = self.class_net(out.view([out.shape[0], -1]))

        return out, att