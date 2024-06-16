import torch.nn as nn
from MLP import MLP
import torch.nn.functional as F
import torch

'''
MMOE的整体实现（为了简化模型结构，这里设置不同专家网络，不同门控网络，不同塔网络的结构一致）

网络结构：
        专家层
        门控层
        塔层（输入是不同专家层输出的加权平均和，不同塔层对应着不同任务的训练）
激活函数：输出层视不同任务而定（本次用的sigmoid函数），门控层输出使用softmax激活函数，其他均为relu
损失函数：视不同任务而定（本次用的交叉熵损失函数）
'''

class MMOE(nn.Module):

    def __init__(self, num_tasks, num_experts, experts_hidden_units, towers_hidden_units, gates_hidden_units):

        '''
        experts_hidden_units： 专家网络每一层的神经元数量
        num_experts: 专家网络的数量
        towers_hidden_units：多塔网络每一层的神经元数量
        num_tasks：训练任务的数量
        gates_hidden_units：门控网络每一层的神经元数量
        '''

        super(MMOE, self).__init__()
        self.experts_hidden_units = experts_hidden_units
        self.num_experts = num_experts
        self.towers_hidden_units = towers_hidden_units
        self.num_tasks = num_experts
        self.gates_hidden_units = gates_hidden_units
        self.num_tasks = num_tasks

        # 专家网络的实现
        self.experts_network  = nn.ModuleList([MLP(self.experts_hidden_units) for i in range(self.num_experts)])
        # 门控网络的实现
        self.gates_network = nn.ModuleList([MLP(self.gates_hidden_units) for i in range(self.num_tasks)])
        # 多塔网络的实现
        self.towers_network = nn.ModuleList([MLP(self.towers_hidden_units) for i in range(self.num_tasks)])
        # 输出层
        self.combination_linear = nn.Linear(self.towers_hidden_units[-1], 1)

    def forward(self, x):
        experts_network_output = [expert(x) for expert in self.experts_network]
        gates_network_output = [F.softmax(gate(x), dim = 1) for gate in self.gates_network]

        final_outputs = []
        for task_idx in range(self.num_tasks):
            weighted_expert_output = torch.stack([gates_network_output[task_idx][:,i].unsqueeze(1) * experts_network_output[i]
                                                  for i in range(self.num_experts)], dim=-1).sum(dim=-1)
            tower_expert_output = self.towers_network[task_idx](weighted_expert_output)
            final_output = F.sigmoid(self.combination_linear(tower_expert_output))
            final_outputs.append(final_output)

        return final_outputs
