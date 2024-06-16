import torch.nn as nn
import torch.nn.functional as F

'''
MLP network的实现

网络结构：全连接线性层
激活函数：Relu
'''
class MLP(nn.Module):

    '''
    linear_layers_units：每一层的神经元个数
    '''

    def __init__(self, linear_layers_units):
        super(MLP, self).__init__()
        self.MLP = nn.ModuleList(
            [nn.Linear(linear_layers_units[index], linear_layers_units[index + 1]) for index in
             range(len(linear_layers_units) - 1)])

    def forward(self, x):
        for linear_layer in self.MLP:
            x = linear_layer(x)
            x = F.relu(x)

        return x
