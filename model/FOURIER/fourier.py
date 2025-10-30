import os

import numpy as np
import pandas as pd
import torch.optim as optim
import torch
from numpy import split

from torch import nn
from model.base_model.base_models import TCN
from matplotlib import pyplot as plt
from matplotlib import font_manager


class fourierTrendSeasonality(nn.Module):
    def __init__(self,args):
        super(fourierTrendSeasonality, self).__init__()
        self.args = args

    def forward(self, x):
        # n_val, seq_len = x.shape
        seq_len,n_val=x.shape
        data = x

        # 将 NumPy 数组转换为 PyTorch Tensor
        tensor_data = data.clone().detach()

        x_fft = torch.fft.fft(tensor_data, dim=0)
        # print('x_fft:', x_fft.shape)
        freq_indices = torch.arange(seq_len)
        high_freq_cutoff = int(seq_len // self.args.fourier_order)  # 根据情况调整，定义趋势性和季节性的分界点

        trend_fft = x_fft.clone()
        trend_fft[ high_freq_cutoff:] = 0  # 保留低频成分作为趋势性

        seasonality_fft = x_fft.clone()
        seasonality_fft[ :high_freq_cutoff] = 0  # 保留高频成分作为季节性

        # 3. 逆傅里叶变换
        trend = torch.fft.ifft(trend_fft, dim=0).real  # 提取趋势性
        seasonality = torch.fft.ifft(seasonality_fft, dim=0).real  # 提取季节性
        return trend, seasonality
    def separate(self,x):
        seq_len,n_val=x.shape
        data = x
        # 将 NumPy 数组转换为 PyTorch Tensor
        tensor_data = data.clone().detach()
        x_fft = torch.fft.fft(tensor_data, dim=0)
        top=0
        end=0
        block=seq_len/self.args.num_layers
        x_fft_list=[]
        for i in range(self.args.num_layers-1):
            end=int((i+1)*block)
            fft_temp=x_fft.clone()
            fft_temp[0:top,:] = 0
            fft_temp[end:,:] = 0
            fft_temp=torch.fft.ifft(fft_temp,dim=0).real
            x_fft_list.append(fft_temp)
            top=end
        fft_temp=x_fft.clone()
        fft_temp[0:end,:] = 0
        fft_temp=torch.fft.ifft(fft_temp,dim=0).real
        x_fft_list.append(fft_temp)
        x_fft_list=torch.stack(x_fft_list,dim=0)
        return x_fft_list#(layers,batch,seq_len,n_val)


def datax_div(x, batch_size,out_length):
    t = len(x)
    size = x.shape[1]
    y = []
    for i in range(t - batch_size-out_length):
        y1 = x.iloc[i:i + batch_size]
        # y=pd.concat([y,y1],ignore_index=True)
        y.append(y1)

    y = pd.concat((y))
    y = np.array(y)
    # print('y:',y.shape,'\nsize:',size,'\nx.shape:',x.shape)
    y = y.reshape(-1, batch_size, size)
    # print('y:', y.shape, '\nsize:', size, '\nx.shape:', x.shape)
    return y


def datay_div(x, batch_size,out_length):
    t = len(x)
    size = x.shape[1]
    y = []
    for i in range(t - batch_size - out_length):
        y1 = x.iloc[i+batch_size:i + out_length+batch_size]
        # y=pd.concat([y,y1],ignore_index=True)
        y.append(y1)

    y = pd.concat((y))
    y = np.array(y)
    # print('y:',y.shape,'\nsize:',size,'\nx.shape:',x.shape)
    y = y.reshape(-1, out_length)
    # print('y:', y.shape, '\nsize:', size, '\nx.shape:', x.shape)
    return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(x, y):
    # 将 NumPy 数组转换为 PyTorch Tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # 将 Tensor 迁移到 GPU
    x = x_tensor.to(device)

    # 将 NumPy 数组转换为 PyTorch Tensor
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 将 Tensor 迁移到 GPU
    y = y_tensor.to(device)
    return x, y


class fourier(nn.Module):
    def __init__(self, num_inputs=4, num_channels=[4, 8,16], batch_size=24, output_length=1):
        super(fourier, self).__init__()
        self._opt = None
        self.num_channels = num_channels
        self.trendSeasonal = fourierTrendSeasonality()
        self.tcn1 = TCN(num_inputs, num_channels, kernel_size=4, output_length=output_length)
        self.tcn2 = TCN(num_inputs, num_channels, kernel_size=4, output_length=output_length)
        self._loss = None
        self.batch_size = batch_size
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.out_length=output_length

    def setup_optimizer(self):
        self._opt = optim.Adam(self.parameters(), lr=0.001)
        self._loss = nn.MSELoss()

    def fit(self, x, y, epochs=100, batch_size=24):
        self.batch_size = batch_size
        self.train()
        # predict=self(x)
        min_train_loss = float('inf')
        # x = x.to(device)
        # y = y.to(device)

        for epoch in range(epochs):
            # print('x:',x.shape)
            x_train = datax_div(x, batch_size,self.out_length)
            y_train = datay_div(y, batch_size,self.out_length)
            # print('y_train:', y_train.shape)
            x_train = np.transpose(x_train, (0, 2, 1))
            # y_train = np.transpose(y_train, (1, 0))
            # 打印数组的形状
            # print('x_train:',x_train.shape)
            print('-------------------epoch=', epoch, '-------------------')
            sum_loss = 0.0
            for batch_id in (range(len(x_train) - 5)):
                batch_x = np.array(x_train[batch_id:batch_id + 5, :, :])
                batch_y = np.array(y_train[batch_id:batch_id + 5, :])
                batch_x, batch_y = to_device(batch_x, batch_y)
                # print(batch_x.shape, batch_y.shape)
                self._opt.zero_grad()
                forcast = self(batch_x)
                forcast = forcast.reshape(-1, self.out_length)
                # print('forcast:',forcast.shape)
                # batch_y = torch.from_numpy(batch_y).float()
                # print('forcast:',forcast.shape,'batch_y:',batch_y.shape)
                loss = self._loss(forcast, batch_y)
                # print('batch_id=',batch_id,'\t loss=',loss)
                sum_loss += loss.item()  # *len(batch_x)
                loss.backward()
                self._opt.step()
            print('loss=', sum_loss / 5)
            if sum_loss < min_train_loss:
                min_train_loss = sum_loss

    def forward(self, x):

        trend, seasonal = self.trendSeasonal(x)
        # print("trend:",trend.shape,"\nseasonal:",seasonal.shape)
        trend = self.tcn1(trend)
        seasonal = self.tcn2(seasonal)
        # trend = torch.relu(self.linear1(trend))
        # seasonal = torch.relu(self.linear2(seasonal))
        x = 0.4*trend + 0.6*seasonal
        # x=self.linear1(x)


        # x=self.tcn1(x)
        # x=x.reshape(-1,1,1)
        # trend,seasonal=self.trendSeasonal(x)
        # trend = torch.relu(self.linear1(trend))
        # seasonal = torch.relu(self.linear2(seasonal))
        # x=trend+seasonal
        return x

    def predict(self, x):

        self.eval()
        # x = datax_div(x, 24)
        x = np.array(x)
        t = len(x)
        # x=x[:t-self.batch_size]
        length, n_val = x.shape
        x = x.reshape(1, length, n_val)
        x = np.transpose(x, (0, 2, 1))
        predict = []
        for i in range(length - self.batch_size-self.out_length):
            x_input = x[:, :, i:i + self.batch_size]
            x_input, _ = to_device(x_input, x_input)
            predict.append(self.forward(x_input).cpu())
        # y = pd.concat((predict))
        # y = np.array(y)

        with torch.no_grad():
            predict = np.array(predict)
        predict = predict.reshape(-1, 1)
        return predict


def mse_mpe(predict, y_test, batch_size=24,output_length=1):
    y_test=datay_div(y_test,batch_size,output_length)
    # print(f'predict.shape:{predict.shape}\ntrue.shape:{y_test.shape}')
    # y_test = y_test[batch_size:]
    predict=predict.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()
    y_test = np.array(y_test)
    # 过滤掉 y_train 中为 0 的数据，避免除以 0
    non_zero_mask = y_test != 0
    y_train_filtered = y_test[non_zero_mask]
    predict_filtered = predict[non_zero_mask]

    # 计算 MSE 和 MPE
    mse = np.mean((y_test - predict) ** 2)
    mpe = np.mean((y_train_filtered - predict_filtered) / y_train_filtered) * 100
    mae = np.mean(np.abs(predict - y_test))
    plot_predict_test(predict, y_test,mpe,mae)
    return mse, mpe, mae


def plot_predict_test(predict, y_test,mpe,mse):
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\MSYHL.TTC")
    plt.figure(figsize=(10, 6))

    # 绘制第一个折线图
    plt.plot(predict, label='predict', color='blue')
    # 绘制第二个折线图
    plt.plot(y_test, label='true', color='red')
    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title(f'predict_test:mpe:{mpe:.2f}  mse:{mse:.2f}')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示图形
    plt.show()

    return 0
