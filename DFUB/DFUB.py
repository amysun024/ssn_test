import torch
import torch.nn as nn
import torch.nn.functional as F

class DFUB(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(DFUB, self).__init__()
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.num_heads = num_heads  # 多头注意力机制的头数

        # 定义线性变换矩阵，用于计算Query, Key, Value
        self.W_Q = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim * num_heads, bias=False)

        # 输出的线性变换矩阵
        self.W_out = nn.Linear(embedding_dim * num_heads, embedding_dim, bias=False)

        # 领域特征和目标物品特征的线性变换矩阵
        self.W_td = nn.Linear(embedding_dim * 2, embedding_dim * num_heads, bias=False)

    def forward(self, E_h, E_t, E_d):
        """
        前向传播函数
        :param E_h: 用户历史行为的嵌入表示, 维度为 (batch_size, seq_len, embedding_dim)
        :param E_t: 目标物品的嵌入表示, 维度为 (batch_size, embedding_dim)
        :param E_d: 领域特征的嵌入表示, 维度为 (batch_size, embedding_dim)
        """
        batch_size, seq_len, _ = E_h.size()

        # 将目标物品和领域特征的嵌入拼接在一起
        E_td = torch.cat([E_t, E_d], dim=-1)  # (batch_size, 2 * embedding_dim)
        E_td = self.W_td(E_td)  # (batch_size, embedding_dim * num_heads)

        # 通过线性变换得到Q, K, V矩阵
        Q = self.W_Q(E_h).view(batch_size, seq_len, self.num_heads,
                               self.embedding_dim)  # (batch_size, seq_len, num_heads, embedding_dim)
        K = self.W_K(E_h).view(batch_size, seq_len, self.num_heads,
                               self.embedding_dim)  # (batch_size, seq_len, num_heads, embedding_dim)
        V = self.W_V(E_h).view(batch_size, seq_len, self.num_heads,
                               self.embedding_dim)  # (batch_size, seq_len, num_heads, embedding_dim)

        # 将领域特征和目标物品特征嵌入转置并与Q, K, V相乘
        E_td = E_td.view(batch_size, self.num_heads, self.embedding_dim).unsqueeze(
            1)  # (batch_size, 1, num_heads, embedding_dim)

        Q = Q * E_td  # (batch_size, seq_len, num_heads, embedding_dim)
        K = K * E_td  # (batch_size, seq_len, num_heads, embedding_dim)
        V = V * E_td  # (batch_size, seq_len, num_heads, embedding_dim)

        # 调整维度以进行矩阵乘法
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, embedding_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, embedding_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, embedding_dim)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                    self.embedding_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, embedding_dim)

        # 将多头注意力的输出拼接在一起
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len,
                                                                                  -1)  # (batch_size, seq_len, num_heads * embedding_dim)

        # 通过输出线性变换得到最终输出
        output = self.W_out(attention_output)  # (batch_size, seq_len, embedding_dim)

        return output

# 示例用法
batch_size = 32
seq_len = 10
embedding_dim = 64
num_heads = 4

# 创建DFUB模块
dfub = DFUB(embedding_dim, num_heads)

# 随机生成输入数据
E_h = torch.randn(batch_size, seq_len, embedding_dim)
E_t = torch.randn(batch_size, embedding_dim)
E_d = torch.randn(batch_size, embedding_dim)

# 前向传播
output = dfub(E_h, E_t, E_d)
print(output.size())  # 应输出 (batch_size, seq_len, embedding_dim)
