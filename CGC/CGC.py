import torch.nn as nn
import torch.nn.functional as F
import torch
from MLP import MLP

'''
CGC的整体实现（为了简化模型结构，这里设置不同专家网络，不同门控网络，不同塔网络的结构一致）

网络结构：
        专家层（task-specific专家网络模块+task-shared专家网络模块）
        门控层（单个隐藏层的MLP）
        塔层（输入是不同专家层输出的加权平均和，不同塔层对应着不同任务的训练）
激活函数：输出层视不同任务而定（本次用的sigmoid函数），门控层输出使用softmax激活函数，其他均为relu
损失函数：视不同任务而定（本次用的交叉熵损失函数，两个二分类任务）
'''
class CGC(nn.Module):

    '''
    experts_hidden_units： 专家网络每一层的神经元数量
    num_shared_experts：共享专家网络的数量
    num_specific_experts：专属专家网络的数量
    towers_hidden_units：多塔网络每一层的神经元数量
    num_tasks：训练任务的数量
    gates_hidden_units: 门控网络每一层的神经元数量
    '''

    def __init__(self, num_specific_experts, num_shared_experts, num_tasks, experts_hidden_units, towers_hidden_units, gates_hidden_units):
        super(CGC, self).__init__()

        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_hidden_units = experts_hidden_units
        self.towers_hidden_units = towers_hidden_units
        self.gates_hidden_units = gates_hidden_units
        self.num_tasks = num_tasks

        # 专家网络的实现
        self.experts_shared = nn.ModuleList([MLP(self.experts_hidden_units) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList([MLP(self.experts_hidden_units) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList([MLP(self.experts_hidden_units) for i in range(self.num_specific_experts)])

        # 门控网络的实现
        self.gates_network = nn.ModuleList([MLP(self.gates_hidden_units) for i in range(self.num_tasks)])
        # 多塔网络的实现
        self.towers_network = nn.ModuleList([MLP(self.towers_hidden_units) for i in range(self.num_tasks)])
        # 输出层
        self.combination_linear = nn.Linear(self.towers_hidden_units[-1], 1)

    def forward(self, x):
        # 计算共享专家的输出
        shared_experts_outputs = [expert(x) for expert in self.experts_shared]
        # 计算任务1专家的输出
        task1_experts_outputs = [expert(x) for expert in self.experts_task1]
        # 计算任务2专家的输出
        task2_experts_outputs = [expert(x) for expert in self.experts_task2]

        # 获取任务1的门控权重及输出
        task1_gate_outputs = F.softmax(self.gates_network[0](x), dim=1)
        task1_combined_output = torch.stack([task1_gate_outputs[:,i].unsqueeze(1) * task1_experts_outputs[i]
                                                  for i in range(self.num_specific_experts)] +
                                            [task1_gate_outputs[:,i].unsqueeze(1) * shared_experts_outputs[i]
                                             for i in range(self.num_specific_experts,)]
                                            , dim=-1).sum(dim=-1)

        # 获取任务2的门控权重及输出
        task2_gate_outputs = F.softmax(self.gates_network[1](x), dim=1)
        task2_combined_output = torch.stack([task2_gate_outputs[:, i].unsqueeze(1) * task2_experts_outputs[i]
                                             for i in range(self.num_specific_experts)] +
                                            [task2_gate_outputs[:, i].unsqueeze(1) * shared_experts_outputs[i]
                                             for i in range(self.num_specific_experts, )]
                                            , dim=-1).sum(dim=-1)

        # 计算每个任务的最终输出
        task1_output = F.sigmoid(self.combination_linear(self.towers_network[0](task1_combined_output)))
        task2_output = F.sigmoid(self.combination_linear(self.towers_network[1](task2_combined_output)))

        return [task1_output, task2_output]