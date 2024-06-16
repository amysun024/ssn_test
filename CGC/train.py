import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from CGC import CGC
import pandas as pd

def data_preparation():
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    train_df = pd.read_csv('data/census-income.data.gz', delimiter=',', header=None, index_col=None, names=column_names)
    test_df = pd.read_csv('data/census-income.test.gz', delimiter=',', header=None, index_col=None, names=column_names)

    # 论文中第一组任务
    label_columns = ['income_50k', 'marital_stat']

    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']

    # 对类别向量进行one-hot编码
    train_transformed = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    test_transformed = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)
    train_labels = train_df[label_columns]
    test_labels = test_df[label_columns]

    test_transformed['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # 转换为二分类问题
    train_income = torch.tensor((train_labels.income_50k == ' 50000+.').astype(int))
    train_marital = torch.tensor((train_labels.marital_stat == ' Never married').astype(int))
    test_income = torch.tensor((test_labels.income_50k == ' 50000+.').astype(int))
    test_marital = torch.tensor((test_labels.marital_stat == ' Never married').astype(int))

    test_data = test_transformed
    test_label = [test_income, test_marital]
    train_data = train_transformed
    train_label = [train_income, train_marital]

    return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = data_preparation()

# 转换为张量
train_tensor_data = TensorDataset(torch.from_numpy(np.array(train_data, dtype=np.float32)), torch.from_numpy(np.array(train_label, dtype=np.float32).T))
train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=1024)

test_tensor_data = TensorDataset(torch.from_numpy(np.array(test_data, dtype=np.float32)), torch.from_numpy(np.array(test_label, dtype=np.float32).T))
test_loader = DataLoader(test_tensor_data, batch_size=1024)

# 定义模型和损失函数，优化器
model = CGC(num_specific_experts=2, num_shared_experts=2, num_tasks=2, experts_hidden_units=[499,64,32], towers_hidden_units=[32,16],
            gates_hidden_units=[499, 2+2])
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    total_loss_epoch = 0.0
    model.train()
    for x, y in train_loader:
        y_hat = model(x)

        y1, y2 = y[:, 0].unsqueeze(1), y[:, 1].unsqueeze(1)
        y_hat1, y_hat2 = y_hat[0], y_hat[1]

        loss1 = loss_fn(y_hat1, y1)
        loss2 = loss_fn(y_hat2, y2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss_epoch += loss.item()
    print('epoch/epoches: {}/{}, train loss: {:.3f}'.format(epoch, 10, total_loss_epoch))