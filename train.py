import numpy as np
import random
import torch.nn
import argparse
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from utils import set_seed
from model import VAE
from dataloader import load_data
from pytorchtools import EarlyStopping

# ============== 系统参数设置 =============== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using {device} in torch")
set_seed(100)
random_state_seed = random.randint(0, 1000)

# ============== 数据、网络参数设置 =============== 
atlas = 'aal'
# atlas_list = ['aal', 'cc200', 'dosenbach160']
hidden_layer_size = 256
n_folds = 5

# ============== 评价指标借口 =================#
loss_ce = torch.nn.CrossEntropyLoss()
loss_mse = torch.nn.MSELoss()

def softmax_pipeline(input):
    pred = softmax(input, axis=1)
    pred = (pred == pred.max(axis=1, keepdims=1)).astype(int)[:, 1]
    return pred

# ============== 加载数据  数据分割=============== #
print(f'Loading dataset {atlas}')
features, labels = load_data(atlas=atlas)
train_valid_data, test_data, train_valid_label, test_label = train_test_split(features, labels, test_size=0.2)
# 给定random_state_seed的情况下，每一次运行的划分都是一致的
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state_seed)
skf_splits = list(skf.split(train_valid_data, train_valid_label))


# ============== n-folds 的数据 =============== #
train_acc_lists = [] 
val_acc_lists = []
patience = 20

test_pred_list = None
# ============== 创建模型 =============== #

for fold in range(n_folds):
    train_acc_list = []
    val_acc_list = []
    kl_list = []
    recf_list = []
    
    best_valid_acc = 0
    
    train_idx, valid_idx = skf_splits[fold]

    train_data = torch.tensor(train_valid_data[train_idx], dtype=torch.float32).to(device)
    valid_data = torch.tensor(train_valid_data[valid_idx], dtype=torch.float32).to(device)
    train_label = torch.tensor(train_valid_label[train_idx], dtype=torch.long).to(device)
    valid_label = torch.tensor(train_valid_label[valid_idx], dtype=torch.long).to(device)

    Model = VAE(input_dim=train_data.size()[1], nhid=16, 
              nclass=2, dropout=0.3, hidden_layer_size=hidden_layer_size)
    Model.to(device)
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.00005, weight_decay=5e-5)

    for epoch in range(400):
        Model.train()  # 启用batch normalization和drop out
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            train_output, rec_f_loss, kl = Model(train_data)
            ce_loss = loss_ce(train_output, train_label)
            loss = ce_loss + kl + rec_f_loss
            loss.backward()
            optimizer.step()

        Model.eval()
        train_pred = softmax_pipeline(train_output.detach().cpu().numpy())
        train_acc = accuracy_score(train_label.detach().cpu().numpy(), train_pred)
        
        with torch.set_grad_enabled(False):  # 不在训练模式了，令计算图不累计梯度
            valid_output, valid_rec_f_loss, valid_kl = Model(valid_data)
            
            valid_ce_loss = loss_ce(valid_output, valid_label)
            valid_loss = valid_ce_loss + valid_rec_f_loss + valid_kl
            
            valid_pred = softmax_pipeline(valid_output.detach().cpu().numpy())
            valid_acc = accuracy_score(valid_label.detach().cpu().numpy(), valid_pred)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            test_output, _, _ = Model(test_data)
            test_pred = softmax_pipeline(test_output.detach().cpu().numpy())
        
        train_acc_list.append(train_acc)
        val_acc_list.append(valid_acc)
        kl_list.append(kl.item())
        recf_list.append(rec_f_loss.item())
        print("KL 散度: {:.4f}, rec_f_loss: {:.4f}".format(
                kl.item(), rec_f_loss.item()), end=" ") 
        print(f"train_acc:{train_acc} valid_acc:{valid_acc}")
    
    if test_pred_list is None:
        test_pred_list = test_pred
    else:
        test_pred_list = np.vstack((test_pred_list, test_pred))
    
    train_acc_lists.append(train_acc_list)
    val_acc_lists.append(val_acc_list)

np.save(f"./result/test_{atlas}.npy", test_pred_list)

