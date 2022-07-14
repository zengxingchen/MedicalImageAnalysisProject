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
from utils import Metric
from torchsummary import summary

# ============== 系统参数设置 =============== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using {device} in torch")
set_seed(100)
random_state_seed = random.randint(0, 1000)

# ============== 解析命令行参数 =============== #
parser = argparse.ArgumentParser()
parser.add_argument('--atlas', type=str, help='the name of atlas')
parser.add_argument('--hidden_layer_size', type=int, help='the size of hidden_layer')
args = parser.parse_args()
atlas = args.atlas
hidden_layer_size = args.hidden_layer_size
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

# 给定random_state_seed的情况下，每一次运行的划分都是一致的

features, labels = load_data(atlas=atlas)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state_seed)
skf_splits = list(skf.split(features, labels))

test_idx = []
for i in range(n_folds):
  test_idx += skf_splits[i][1].tolist()
true_labels = labels[test_idx]
np.save(r'true_labels.npy', np.array(true_labels))

# train_valid_data, test_data, train_valid_label, test_label = train_test_split(features, labels, test_size=0.2)

# ============== n-folds 的数据 =============== #
train_acc_lists = [] 
val_acc_lists = []
all_pred = []

# ============== 创建模型 =============== #
for fold in range(n_folds):
    train_acc_list = []
    val_acc_list = []
    kl_list = []
    recf_list = []
    
    best_valid_acc = 0
    
    train_valid_idx, test_idx = skf_splits[fold]
    
    train_data, valid_data, train_label, valid_label = train_test_split(features[train_valid_idx], labels[train_valid_idx], test_size=0.1)
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    valid_data = torch.tensor(valid_data, dtype=torch.float32).to(device)
    test_data = torch.tensor(features[test_idx], dtype=torch.float32).to(device)
    train_label = torch.tensor(train_label, dtype=torch.long).to(device)
    valid_label = torch.tensor(valid_label, dtype=torch.long).to(device)
    test_label = torch.tensor(labels[test_idx], dtype=torch.long).to(device)

    Model = MLP(input_dim=train_data.size()[1], nhid=16, 
              nclass=2, dropout=0.3, hidden_layer_size=hidden_layer_size)
    Model.to(device)
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.00005, weight_decay=5e-5)

    for epoch in range(400):
        Model.train()  # 启用batch normalization和drop out
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            train_output  = Model(train_data)
            loss = loss_ce(train_output, train_label)
            loss.backward()
            optimizer.step()

        Model.eval()
        train_pred = softmax_pipeline(train_output.detach().cpu().numpy())
        train_acc = accuracy_score(train_label.detach().cpu().numpy(), train_pred)
        
        with torch.set_grad_enabled(False):  # 不在训练模式了，令计算图不累计梯度
            valid_output = Model(valid_data)
            
            valid_loss = loss_ce(valid_output, valid_label)

            
            valid_pred = softmax_pipeline(valid_output.detach().cpu().numpy())
            valid_acc = accuracy_score(valid_label.detach().cpu().numpy(), valid_pred)
            
            test_output = Model(test_data)
            test_loss = loss_ce(test_output, test_label)
            
            test_pred = softmax_pipeline(test_output.detach().cpu().numpy())
            test_acc = accuracy_score(test_label.detach().cpu().numpy(), test_pred)
        
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_test_pred = test_pred

        
        print('-'*89)
        print('| end of epoch {} | valid loss {:3.2f} | train acc {:4.2f}'
              'valid_acc {:4.2f} | test_acc {:4.2f}'.format(epoch, valid_loss, train_acc, valid_acc, test_acc)
              )
        
        train_acc_list.append(train_acc)
        val_acc_list.append(valid_acc)
        # print("KL 散度: {:.4f}, rec_f_loss: {:.4f}".format(
        #         kl.item(), rec_f_loss.item()), end=" ") 
        # print(f"train_acc:{train_acc} valid_acc:{valid_acc}")
    
    all_pred += best_test_pred.tolist()

    train_acc_lists.append(train_acc_list)
    val_acc_lists.append(val_acc_list)

final_acc = accuracy_score(true_labels, all_pred)
print('-'*89)
print('| End of training | acc  {:6.2f}'.format(final_acc)
    )
# np.save(f"./result/test_{atlas}.npy", np.array(all_pred))


result = Metric(np.array(true_labels), np.array(all_pred), soft=True, dim=1, datatype="numpy")

with open('lab_records.txt', 'a') as f:

  f.write(f"================{atlas}================\n")
  f.write("ACC: {:.2f}%\n".format(result[0] * 100))
  f.write("AUC: {:.2f}%\n".format(result[1] * 100))
  f.write("SEN: {:.2f}%\n".format(result[2] * 100))
  f.write("SPE: {:.2f}%\n".format(result[3] * 100))
  f.write("F1: {:.2f}%\n".format(result[4] * 100))
