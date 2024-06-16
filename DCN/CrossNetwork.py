import torch.nn as nn
import torch

'''
cross network的实现

网络结构：每一层的输出等于初始输入，特定层的权重矩阵和前一层输出的乘积，加上偏置和前一层的输出
激活函数：没有显式使用激活函数
'''
class CrossNetwork(nn.Module):

    '''
    layer_num：网络层数
    input_dim：输入向量的维度
    '''

    def __init__(self, input_dim, num_layers):

        super(CrossNetwork, self).__init__()

        self.num_layers = num_layers
        self.cross_layers = nn.ParameterList([nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)])
        # 采用xavier初始化
        for cross_layer in self.cross_layers :
            nn.init.xavier_normal_(cross_layer)
        self.bias_layers = nn.ParameterList([nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)])

    def forward(self, x):
        x_0 = x.unsqueeze(2)
        x_l = x_0
        for i in range(self.num_layers):
            dot1 = torch.matmul(x_0, x_l.transpose(1,2))
            dot2 = torch.matmul(dot1, self.cross_layers[i])
            x_l = dot2 + self.bias_layers[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l
