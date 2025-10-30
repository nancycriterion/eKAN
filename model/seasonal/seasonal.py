import math
from concurrent.futures import ThreadPoolExecutor

from torch.nn import Dropout

from data.read_data import normalize,denormalize
import numpy as np
import torch
import torch.nn as nn
from model.FOURIER.fourier import *


class sae(nn.Module):
    def __init__(self,input_length,hidden_size,sparsity_ratio,device='cpu'):
        super(sae, self).__init__()
        self.input_length=input_length
        self.hidden_size=hidden_size
        self.sparsity_ratio=sparsity_ratio
        self.device=device
        self.encoder = nn.Linear(input_length, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_length)
        self.to(device)

    def forward(self,x):
        '''
        :param x: (input_length,input_size)
        :return: de:(batchs,input_length,input_size)：解码后的数据
                 en:(batchs,hidden_size,input_size):编码后的数据
                 x:(batchs,input_length,input_size):原始数据
        '''
        x=x.to(self.device)
        #把x形状交换得到(input_size,input_length)
        x=x.permute(1,0)
        length=x.shape[1]
        output_en=[]
        output_de=[]
        output_x=[]
        for i in range(length-self.input_length+1):
            x_temp=x[:,i:i+self.input_length]
            temp_en=self.encoder(x_temp)#(input_size,hidden_size)
            temp_de=self.decoder(temp_en)#(input_size,input_length)
            temp_en=temp_en.permute(1,0)
            output_en.append(temp_en)
            temp_de=temp_de.permute(1,0)
            output_de.append(temp_de)
            x_temp=x_temp.permute(1,0)
            output_x.append(x_temp)
        # output_x=torch.stack(output_x,dim=0)
        output_en=torch.stack(output_en,dim=0)
        # output_de=torch.stack(output_de,dim=0)
        return output_de,output_en,output_x

    def sparse_loss(self, encoded):
        # 稀疏性损失：KL 散度
        epsilon = 1e-10  # 防止数值问题
        sparsity = torch.mean(encoded, dim=0)
        kl_div = self.sparsity_ratio * torch.log(self.sparsity_ratio / (sparsity + epsilon)) + \
                 (1 - self.sparsity_ratio) * torch.log((1 - self.sparsity_ratio) / (1 - sparsity + epsilon))
        return kl_div.sum()
    def en_get(self,x):
        '''
        :param x:
        :return: encoded: 编码后的数据(batchs,hidden_size,input_size)
        '''
        self.eval()
        with torch.no_grad():
            x=x.to(self.device)
            output_de,output_en,output_x=self(x)
        return output_en


class MultiScaleSAE(nn.Module):
    def __init__(self, input_lengths=None, hidden_size=128, sparsity_ratio=0.1, device='cpu'):
        super().__init__()
        if input_lengths is None:
            input_lengths = [64, 32, 16]
        self.scale_encoders = nn.ModuleList([
            sae(input_len, hidden_size, sparsity_ratio, device)
            for input_len in input_lengths
        ])
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * len(input_lengths), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(input_lengths)),
            nn.Softmax(dim=-1)
        )
        self.device = device


    def forward(self, x):
        # 各尺度编码 [B, T, C] -> [B, hidden_size, C]
        encoded_scales = [encoder.en_get(x)[:, -1, :] for encoder in self.scale_encoders]  # 取最后时刻编码

        # 计算注意力权重
        concat_features = torch.cat(encoded_scales, dim=-1)  # [B, hidden_size * num_scales]
        weights = self.attention(concat_features)  # [B, num_scales]

        # 加权融合
        refined = sum(w.unsqueeze(-1) * enc for w, enc in zip(weights.unbind(-1), encoded_scales))  # [B, hidden_size]
        return refined



class HierarchicalSAE(nn.Module):
    def __init__(self, input_length=64, hidden_size=128, sparsity_ratio=0.1, device='cpu'):
        super().__init__()
        self.coarse_encoder = sae(input_length, hidden_size, sparsity_ratio, device)
        self.fine_encoder = sae(input_length // 2, hidden_size, sparsity_ratio, device)
        self.device = device

    def forward(self, x):
        # 粗粒度编码
        coarse_encoded = self.coarse_encoder.en_get(x)  # [B, hidden_size, C]
        coarse_recon = self.coarse_encoder.decoder(coarse_encoded)  # [B, input_length, C]

        # 计算残差（高频部分）
        residual = x[:, -coarse_recon.shape[1]:, :] - coarse_recon  # 对齐时间步

        # 细粒度编码残差
        fine_encoded = self.fine_encoder.en_get(residual)

        # 确保 fine_encoded 的形状与 coarse_encoded 的形状匹配
        # 这里假设 fine_encoded 的形状为 (batchs, hidden_size, input_size // 2)
        batch_size, hidden_size, input_size = coarse_encoded.shape
        fine_encoded = fine_encoded.unsqueeze(-1).expand(batch_size, hidden_size, input_size)

        return torch.cat([coarse_encoded, fine_encoded], dim=-2)  # 拼接多尺度特征


class EnhancedSAE(nn.Module):
    def __init__(self, input_length=64, hidden_size=128, sparsity_ratio=0.1, device='cpu'):
        super().__init__()
        # 原始SAE
        self.base_sae = sae(input_length, hidden_size, sparsity_ratio, device)

        # 多尺度扩展
        self.multi_scale = MultiScaleSAE(
            input_lengths=[input_length, input_length // 2, input_length // 4],
            hidden_size=hidden_size,
            sparsity_ratio=sparsity_ratio,
            device=device
        )

        # 层次化分解
        self.hierarchical = HierarchicalSAE(
            input_length=input_length,
            hidden_size=hidden_size,
            device=device
        )

    def forward(self, x):
        # 原始编码
        base_de, base_en, base_x = self.base_sae(x)

        # 多尺度特征
        # print('多尺度特征输入形状',x.shape,end=' ')
        ms_feat = self.multi_scale(x)

        # 层次化特征
        hier_feat = self.hierarchical(x)

        # 特征融合
        batch_size, hidden_size, input_size = base_en.shape
        ms_feat = ms_feat.unsqueeze(-1).expand(batch_size, hidden_size, input_size)
        hier_feat = hier_feat.unsqueeze(-1).expand(batch_size, hidden_size, input_size)

        combined = torch.cat([base_en, ms_feat, hier_feat], dim=-2)

        return combined

    def en_get(self, x):
        return self.base_sae.en_get(x)

class STkan(nn.Module):
    def __init__(self, in_features, length, out_features, num_frequencies=10, scale_base=1.0, scale_fourier=1.0,
                 base_activate=nn.SiLU, device='cpu'):
        '''
        '''
        super(STkan, self).__init__()
        self.in_features = in_features
        self.length = length
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier
        self.base_activate = base_activate()
        self.fourier_coeffs = nn.Parameter(torch.Tensor(2, out_features, length, in_features, num_frequencies))
        self.reset_parameters()
        self.device = device
        self.to(device)

    def reset_parameters(self):
        '''初始化'''
        with torch.no_grad():
            frequency_decay = torch.ones(self.num_frequencies)
            # 计算标准差(num_frequencies,)
            std = self.scale_fourier / math.sqrt(self.in_features) / frequency_decay
            std = std.view(1, 1, 1, -1)
            self.fourier_coeffs[0].uniform_(-1, 1)
            self.fourier_coeffs[1].uniform_(-1, 1)
            self.fourier_coeffs[0].mul_(std)
            self.fourier_coeffs[1].mul_(std)

    def forward(self, x):
        '''
        输入shape:(batch_size,length,in_features)
        输出shape:(batch_size,length,out_features)
        '''
        original_shape = x.shape
        # 获取傅里叶频率 (1, 1,1, num_frequencies)
        k = torch.arange(1, self.num_frequencies + 1, device=self.device).view(1, 1, 1, -1)
        # (batch_size,length,in_features,1)
        x_expanded = x.unsqueeze(-1)
        xk = x_expanded * k  # (batch_size,length,in_features,num_frequencies)
        # print(x.shape,x_expanded.shape, xk.shape)
        # 正弦余弦
        cos_xk = torch.cos(xk)
        sin_xk = torch.sin(xk)
        # 计算输出
        # print(cos_xk.shape, sin_xk.shape, self.fourier_coeffs.shape)
        cos_part = torch.einsum('blin,olin->blo', cos_xk, self.fourier_coeffs[0])
        sin_part = torch.einsum('blin,olin->blo', sin_xk, self.fourier_coeffs[1])
        output = cos_part + sin_part
        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算傅里叶系数的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            torch.Tensor: 正则化损失值。
        """
        coeffs_l2 = self.fourier_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class mul_kan_model(nn.Module):
    def __init__(self, layers_hidden, length,args=None, num_frequencies=10, scale_base=1.0, scale_fourier=1.0,
                 base_activation=nn.SiLU, device='cpu'):
        '''

        :param layers_hidden: 每一层的输入和输出维度
        :param length:
        :param num_frequencies:
        :param scale_base:
        :param scale_fourier:
        :param base_activation:
        :param device:
        '''
        super(mul_kan_model, self).__init__()
        self.layers = nn.ModuleList()
        self.length = length
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(STkan(in_features=in_features, length=length, out_features=out_features,
                                     num_frequencies=num_frequencies,
                                     scale_base=scale_base, scale_fourier=scale_fourier, base_activate=base_activation,
                                     device=device))

    def forward(self, x: torch.Tensor):
        '''

        :param x: （batch_size,length,in_features）
        :return: x: （batch_size,length,out_features）
        '''
        for layer in self.layers:
            # print('kan层输入形状',x.shape,end=' ')
            x = layer(x)
            # print('kan层输出形状',x.shape)
        return x

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算模型的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_coeffs)
            for layer in self.layers
        )

class sae_kan_model(nn.Module):
    def __init__(self,input_size,input_length,out_length,num_layers:int,lr,device='cpu',args=None):
        '''
        输入：（length,input_size）
        输出：（out_length）
        :param input_size:
        :param input_length:
        :param out_length:
        :param num_layers:
        :param lr:
        :param device:
        '''
        super(sae_kan_model, self).__init__()
        self.fourier_encoder = fourierTrendSeasonality(args)
        self.args = args
        self.batch_size = None
        self.device = device
        self.input_size = input_size
        self.input_length = input_length
        self.num_layers = num_layers
        hidden_size = np.random.choice(range(out_length, self.input_length + 1), size=self.num_layers,
                                       replace=True)  # 随机选择隐藏层的数量
        hidden_size=np.random.choice(range(out_length,out_length+1),size=self.num_layers)
        self.saes_layers = []
        self.kan_layers = []
        self.linear_layers = []
        for i in range(self.num_layers):
            self.saes_layers.append(
                EnhancedSAE(input_length=input_length, hidden_size=int(hidden_size[i]), sparsity_ratio=lr, device=self.device))
            self.kan_layers.append(
                mul_kan_model(layers_hidden=[input_size, 6, 12, 6, 1], length=hidden_size[i], args=args, scale_base=10,
                              base_activation=nn.SiLU, device=self.device))
            self.linear_layers.append(nn.Linear(int(hidden_size[i]), self.args.out_length))
        self.saes_layers = nn.ModuleList(self.saes_layers)
        self.kan_layers = nn.ModuleList(self.kan_layers)
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)
        self.to(self.device)
    '''
    伪代码：
    Input: x, w_cos, w_sin #The input data `x`, the cosine and sine components of the Fourier coefficients `w_cos` and `w_sin`
    Output: out
    Initialize an empty list `out` to store the final results
    for i in range(num_layers):
        encoded_x=x[i].saes_layers[i] #Get the encoded data `encoded_x` from the corresponding SAE layer
        for x in encoded_x:
            for layer in kan_layers;
                cos=torch.cos(x) #Compute the cosine of `x`
                sin=torch.sin(x) #Compute the sine of `x`
                x=cos*w_cos+sin*w_sin #Combine the cosine and sine components
                x.reshape
        out.append(y) #Append the processed `x` to the output list `out`
    return out #Return the final list `out`, which contains the processed outputs from all layers
    
    按时间窗口大小遍历x：
        用SAE层对对一定时间长度窗口内的x进行编码
        对编码后的x进行堆叠兼容kan层的输入形状
        遍历kan层：
        迭代处理x
        x=kan(x)


begin:设置input_length=l_i,batch_size=b
while(还存在未被编码的下):
    以step=1截取长度为b+l-1的x
    for i=0 to b do:
        利用SAE将长为l的x编码为长为sae.en_length的x_en
        计算编码损失梯度优化编码器参数
        （不同的sae_kan层编码后长度随机）
    处理后的x形状为（batch_size,sae.en_length,input_number）
    for j=0 to kan_layers do:
        利用kan层对x_en进行处理
        通过公式（24）获取输入x多个变量之间的相互影响关系
        get y=kan(x_en)
    '''

    def forward1(self, x):
        out_list = []
        batch_size = self.batch_size

        x=self.fourier_encoder.separate(x)
        for i in range(self.num_layers):
            out = []
            # x_train=x[i]
            # np.savetxt(f'D:\\code\\zy\\predict_use\\data\\result\\fourier\\{i}.csv', x_train.cpu().numpy(), delimiter=',')
            # encoded_train = self.saes_layers[i].en_get(x[i])
            encoded_train = self.saes_layers[i](x[i])
            # print('sae层输出：',encoded_train.shape)
            for j in range(len(encoded_train) - batch_size + 1):
                x_train_batch = encoded_train[j:j + batch_size]
                # print(f'{j}kan层输入：', x_train_batch.shape)
                output = self.kan_layers[i](x_train_batch)
                # print(output.shape)
                output = output.reshape(batch_size, -1)

                output = self.linear_layers[i](output)
                # print('kan层输出：', output.shape)
                out.append(output)
            out = torch.stack(out, dim=0)
            out_list.append(out)
        out_list = torch.stack(out_list, dim=0)

        return out_list

    def forward(self, x):
        out_list = []
        batch_size = self.batch_size

        x = self.fourier_encoder.separate(x)
        for i in range(self.num_layers):
            out = []
            encoded_train = self.saes_layers[i].en_get(x[i])
            for j in range(len(encoded_train) - batch_size + 1):
                x_train_batch = encoded_train[j:j + batch_size]
                output = self.kan_layers[i](x_train_batch)
                output = output.reshape(batch_size, -1)
                output = self.linear_layers[i](output)
                output=self.dropout(output)
                out.append(output)
            out = torch.stack(out, dim=0)
            out_list.append(out)
        out_list = torch.stack(out_list, dim=0)

        return out_list

    def fit(self, x_train, y_train, x_val, y_val, epochs_sae, epochs_kan, batch_size):
        self.batch_size=batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.saes_layers.train()
        x_train = self.fourier_encoder.separate(x_train)
        y_train = self.fourier_encoder.separate(y_train)
        x_val = self.fourier_encoder.separate(x_val)
        y_val = self.fourier_encoder.separate(y_val)
        print(':::::::::::::::::::::::sae_train_begining:::::::::::::::::::::::::::')

        for epoch in range(epochs_sae):
            optimizer.zero_grad()
            loss = 0
            for i in range(self.num_layers):
                loss += self.sae_fit(i, x_train[i], criterion)
            loss.backward(retain_graph=True)
            optimizer.step()
            with torch.no_grad():
                val_loss = 0
                for i in range(self.num_layers):
                    decoded_train, encoded_train, x_train_list = self.saes_layers[i](x_val[i])
                    decoded_train = torch.stack(decoded_train, dim=1)
                    x_train_list = torch.stack(x_train_list, dim=1)
                    val_loss += criterion(decoded_train, x_train_list)
                loss = loss / (self.num_layers * len(x_train) * self.input_length)
                val_loss = val_loss / (self.num_layers * len(x_val) * self.input_length)
                '''
                if (epoch + 1) % 5 == 0:
                    print('Epoch: {}/{}..'.format(epoch + 1, epochs_sae),
                          'Loss: {:.4f}..'.format(loss.item()),
                          'Val Loss: {:.4f}..'.format(val_loss.item()))
                '''
        del optimizer,criterion
        self.saes_layers.requires_grad_(False)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        print(':::::::::::::::::::::::mul_kan_train_begining:::::::::::::::::::::::::::')
        for epoch in range(epochs_kan):
            optimizer.zero_grad()
            loss = 0
            for i in range(self.num_layers):
                loss+=self.kan_fit(i,x_train[i],y_train[i],batch_size,criterion)
            optimizer.step()
            with torch.no_grad():
                val_loss = 0
                for i in range(self.num_layers):
                    encoded_val = self.saes_layers[i].en_get(x_val[i])
                    # encoded_val=torch.stack(encoded_val,dim=0)
                    for j in range(len(encoded_val) - batch_size + 1):
                        # print('验证集验证', i, j)
                        x_val_batch = encoded_val[j:j + batch_size]
                        y_val_batch = []
                        for k in range(batch_size):
                            y_val_batch.append(
                                (y_val[i])[self.input_length + j - 1 - self.args.out_length:self.input_length + j - 1])
                        y_val_batch = torch.stack(y_val_batch, dim=0)
                        output = self.kan_layers[i](x_val_batch)
                        output = output.reshape(batch_size, -1)
                        output = self.linear_layers[i](output)
                        # print('验证集验证', i, j, output.shape, y_val_batch.shape)
                        val_loss += criterion(output, y_val_batch.reshape(batch_size, -1).to(self.device))
                if (epoch + 1) % 2 == 0:
                    print('sae_kan_train_Epoch: {}/{}..'.format(epoch + 1, epochs_kan),
                          'Loss: {:.4f}..'.format(loss.item() / batch_size / self.num_layers),
                          'Val Loss: {:.4f}..'.format(val_loss.item() / batch_size / self.num_layers))
        print(':::::::::::::::::::::::mul_kan_train_end:::::::::::::::::::::::::::::::::')
        del optimizer,criterion

    def sae_fit(self, i, x_train, criterion):
        decoded_train, encoded_train, x_train_list = self.saes_layers[i](x_train)
        decoded_train = torch.stack(decoded_train, dim=1)
        x_train_list = torch.stack(x_train_list, dim=1)
        loss = criterion(decoded_train, x_train_list)

        return loss

    def kan_fit(self, i, x_train, y_train, batch_size, criterion):
        encoded_train = self.saes_layers[i].en_get(x_train)
        # encoded_train=torch.stack(encoded_train,dim=0)
        loss = 0
        for j in range(len(encoded_train) - batch_size + 1):
            x_train_batch = encoded_train[j:j + batch_size]
            y_train_batch = []
            for k in range(batch_size):
                y_train_batch.append(
                    y_train[self.input_length + j - 1 - self.args.out_length:self.input_length + j - 1])
            y_train_batch = torch.stack(y_train_batch, dim=0)
            # print('sae输出：',encoded_train.shape,'kan输入：',x_train_batch.shape,'真实值：',y_train_batch.shape)
            # x_train_batch,means,std=normalize(x_train_batch)
            # y_train_batch,means_y,std_y=normalize(y_train_batch)
            output = self.kan_layers[i](x_train_batch)
            # print('堆叠kan输出：',output.shape)
            # output = output[:,-1*self.args.out_length:,:]
            output = self.linear_layers[i](output.reshape(batch_size, -1))
            # output=denormalize(output.reshape(batch_size,self.args.out_length,-1).to('cpu'),means_y,std_y)
            # print('线性层输出：',output.shape,'真实值：',y_train_batch.shape)
            loss += criterion(output.reshape(batch_size, -1).to(self.device), y_train_batch.reshape(batch_size, -1).to(self.device))
            loss.backward(retain_graph=True)
        return loss

    def test(self, x_test):
        out = []
        x_test=self.fourier_encoder.separate(x_test)
        for i in range(self.num_layers):
            out1 = []
            encoded_train = self.saes_layers[i].en_get(x_test[i])
            # print('sae层输出：',encoded_train.shape)
            for j in range(len(encoded_train)):
                x_train_batch = encoded_train[j:j +1]
                # print(f'{j}kan层输入：', x_train_batch.shape)
                output = self.kan_layers[i](x_train_batch)
                output = output.reshape(1, -1)
                output = self.linear_layers[i](output)
                # print('kan层输出：', output.shape)
                out1.append(output)
            out1 = torch.stack(out1, dim=0)
            out.append(out1)
        out = torch.stack(out, dim=0)
        return out
    def predict(self, x_test):
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(self.num_layers):
                out1 = []
                encoded_test = self.saes_layers[i].en_get(x_test)
                for j in range(len(encoded_test)):
                    x_test_batch = encoded_test[j:j + 1]
                    output = self.kan_layers[i](x_test_batch)
                    output = output.reshape(1, -1)
                    output = self.linear_layers[i](output)
                    # output=output.reshape(1,-1)
                    out1.append(output)
                out1 = torch.stack(out1, dim=0)
                out.append(out1)
            out = torch.stack(out, dim=0)
        return out