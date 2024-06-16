from random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from DCN import DCN
import pandas as pd
import torch
import numpy as np

# 读入数据集
sparse_feature = ['C' + str(i) for i in range(1, 27)]
dense_feature = ['I' + str(i) for i in range(1, 14)]
data = pd.read_csv('data/criteo_sample.txt', sep=',')

# 处理缺失值
data[sparse_feature] = data[sparse_feature].fillna('-1', )
data[dense_feature] = data[dense_feature].fillna('0',)
target = ['label']

# 构建特征词典
feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_feature}

# 对连续特征和类别特征分别进行编码
for feat in sparse_feature:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
nms = MinMaxScaler(feature_range=(0, 1))
data[dense_feature] = nms.fit_transform(data[dense_feature])

# 切分数据集
seed = seed(1)
train, test = train_test_split(data, test_size=0.2, random_state=seed)

# 转换为张量
train_label = pd.DataFrame(train['label'])
train = train.drop(columns=['label'])
train_tensor_data = TensorDataset(torch.from_numpy(np.array(train)), torch.from_numpy(np.array(train_label)))
train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=16)

test_label = pd.DataFrame(test['label'])
test = test.drop(columns=['label'])
test_tensor_data = TensorDataset(torch.from_numpy(np.array(test)), torch.from_numpy(np.array(test_label)))
test_loader = DataLoader(test_tensor_data, batch_size=16)

# 定义模型和损失函数，优化器
model = DCN([128, 64], dense_feature, sparse_feature, 2, feat_sizes_sparse)
loss_func = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    total_loss_epoch = 0.0
    model.train()
    for index, (x, y) in enumerate(train_loader):
        y_hat = model(x)
        optimizer.zero_grad()
        loss = loss_func(y_hat.float(), y.float())
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
    print('epoch/epoches: {}/{}, train loss: {:.3f}'.format(epoch, 10, total_loss_epoch))
