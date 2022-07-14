import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, nhid, nclass, dropout, hidden_layer_size):
        super(VAE, self).__init__()
  
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, 2400),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.deconder1 = nn.Sequential(
            nn.Linear(1200, input_dim),
            nn.Sigmoid()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(1200, 600),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )
        self.deconder2 = nn.Sequential(
            nn.Linear(600, 1200),
            nn.Sigmoid()
        )

        # self.encoder3 = nn.Sequential(
        #     nn.Linear(600, 300),
        #     nn.Dropout(p=0.3),
        #     nn.ReLU()
        # )
        # self.deconder3 = nn.Sequential(
        #     nn.Linear(300, 600),
        #     nn.Sigmoid()
        # )
        
        
        self.mse_loss = torch.nn.MSELoss()
        self.MLP = nn.Sequential(
                # torch.nn.Linear(600, hidden_layer_size),
                # torch.nn.ReLU(inplace=True),
                # nn.Dropout(p=0.5),
                # nn.BatchNorm1d(hidden_layer_size*4),
                torch.nn.Linear(600, hidden_layer_size*2),
                nn.Dropout(p=0.5),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_layer_size*2),
                torch.nn.Linear(hidden_layer_size*2, hidden_layer_size),
                nn.Dropout(p=0.5),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_layer_size),
                torch.nn.Linear(hidden_layer_size, nclass))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        encode_feature_1 = self.encoder1(x)
        mu, siama = encode_feature_1.chunk(2, dim=1)
        normal = torch.rand_like(siama)
        encode_feature_1 = mu + siama * normal
        decoder_feature_1 = self.deconder1(encode_feature_1)

        # ======== 求 kl 散度 ==========
        # 预测分布
        log_encode_feature = F.log_softmax(encode_feature_1, dim=-1)
        # 真实分布
        softmax_normal = F.softmax(normal, dim=-1)
        # F.kl_div 第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵。这
        # 里很重要，不然求出来的kl散度可能是个负值。

        # 用  预测分布(log_encode_feature) 拟合 真实分布(softmax_normal) 
        kl = F.kl_div(log_encode_feature, softmax_normal, reduction='sum')

        encode_feature_2 = self.encoder2(encode_feature_1)
        decoder_feature_2 = self.deconder2(encode_feature_2)

        # encode_feature_3 = self.encoder3(encode_feature_2)
        # decoder_feature_3 = self.deconder3(encode_feature_3)
        sum_mse = 0
        sum_mse += self.mse_loss(x, decoder_feature_1)
        sum_mse += self.mse_loss(encode_feature_1, decoder_feature_2)
        # sum_mse += self.mse_loss(encode_feature_2, decoder_feature_3)

        output = self.MLP(encode_feature_2)
        return output, sum_mse, kl



class MLP(nn.Module):
    def __init__(self, input_dim, nhid, nclass, dropout, hidden_layer_size):
        super(MLP, self).__init__()
  
        self.mse_loss = torch.nn.MSELoss()
        self.MLP = nn.Sequential(
                # torch.nn.Linear(600, hidden_layer_size),
                # torch.nn.ReLU(inplace=True),
                # nn.Dropout(p=0.5),
                # nn.BatchNorm1d(hidden_layer_size*4),
                torch.nn.Linear(input_dim, hidden_layer_size*2),
                nn.Dropout(p=0.5),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_layer_size*2),
                torch.nn.Linear(hidden_layer_size*2, hidden_layer_size),
                nn.Dropout(p=0.5),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_layer_size),
                torch.nn.Linear(hidden_layer_size, nclass))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        output = self.MLP(x)
        return output

   