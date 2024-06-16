import torch.nn as nn
import torch.nn.functional as F

'''
deep network的实现

网络结构：全连接线性层
激活函数：Relu
'''
class DeepNetwork(nn.Module):

    '''
    linear_layers_units：每一层的神经元个数
    '''

    def __init__(self, input_dim, linear_layers_units):

        super(DeepNetwork, self).__init__()

        linear_layers_units.insert(0, input_dim)
        self.dnn_network = nn.ModuleList(
            [nn.Linear(linear_layers_units[index], linear_layers_units[index+1]) for index in range(len(linear_layers_units)-1)])

    def forward(self, x):

        for linear_layer in self.dnn_network:
            x = linear_layer(x)
            x = F.relu(x)

        return x

