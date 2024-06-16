import torch
import torch.nn as nn
import torch.nn.functional as F
import DeepNetwork
import CrossNetwork

'''
deep&cross network的整体实现

网络结构：
        特征层（拼接 embedding后的稀疏特征向量 + 稠密特征向量）
        CrossNetwork + DeepNetwork （并行计算）
        结合层（拼接 CrossNetwork + DeepNetwork的输出向量）
激活函数：结合层使用sigmoid激活函数
损失函数：交叉熵损失函数
'''

class DCN(nn.Module):

    '''
    dnn_hidden_units：dnn网络每一层神经元个数
    layer_num：crossNetwork层数
    sparse_feature_cols：稀疏特征向量
    dense_feature_cols：稠密特征向量
    '''

    def __init__(self, dnn_hidden_units, dense_feature_cols, sparse_feature_cols, layer_num, feat_sizes_sparse):
        super(DCN, self).__init__()
        self.dnn_hidden_units = dnn_hidden_units
        self.dense_feature_cols, self.sparse_feature_cols = dense_feature_cols, sparse_feature_cols

        self.embed_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings=feat_sizes_sparse[feature], embedding_dim=10) for feature in self.sparse_feature_cols])
        self.input_dim = len(dense_feature_cols) + 10 * len(sparse_feature_cols)
        self.dnn_network = DeepNetwork.DeepNetwork(self.input_dim, dnn_hidden_units)
        self.cross_network = CrossNetwork.CrossNetwork(dnn_hidden_units[0], layer_num)
        self.combination_linear = nn.Linear(dnn_hidden_units[-1] + dnn_hidden_units[0], 1)

    def forward(self, x):
        dense_x = x[:, :len(self.dense_feature_cols)]  # 密集特征部分
        sparse_x = x[:, len(self.dense_feature_cols): ]  # 稀疏特征部分

        embeds_feature_cols = [self.embed_layers[i](sparse_x[:, i].long()) for i in range(len(self.sparse_feature_cols))]
        embeds = torch.cat(embeds_feature_cols, dim=1)

        # 拼接 embedding后的稀疏特征向量 + 稠密特征向量
        x = torch.cat([dense_x.to(torch.float32), embeds.to(torch.float32)], axis=-1)

        cross_network_output = self.cross_network(x)

        deep_network_output = self.dnn_network(x)

        combination_output = torch.cat([cross_network_output, deep_network_output], axis=-1)

        final_output = F.sigmoid(self.combination_linear(combination_output))

        return final_output

