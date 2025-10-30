import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from .seasonal.seasonal import sae_kan_model
from data.read_data import ploting, normalize3, denormalize3, normalize


class stsakan_no_trend(nn.Module):  # 消融版本类名
    def __init__(self, input_size_t, input_size_s, hidden_size, output_size, input_length, out_length, num_layers,
                 kernel_size, padding, lr, args=None, stride=1, device='cpu'):
        """
        消融版本：移除趋势模块后的模型
        """
        super(stsakan_no_trend, self).__init__()
        self.args = args
        self.input_size_s = input_size_s
        self.out_length = out_length
        self.num_layers = num_layers
        self.lr = lr
        self.device = device

        # 仅保留季节性模块
        self.seasonal = sae_kan_model(
            input_size=input_size_s,
            input_length=args.input_length,
            out_length=out_length,
            num_layers=num_layers,
            lr=lr,
            device=device,
            args=args
        )

        # 移除所有趋势相关参数
        self.linear2 = nn.Linear(num_layers, out_length)  # 用于季节性输出
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, x_s):
        """
        简化后的前向传播：仅使用季节性模块
        """
        # seasonal处理后的形状：(num_layers, batch_length, batch_size, out_length)
        y_s = self.seasonal(x_s)
        y_s = y_s.permute(2, 0, 1, 3)  # (batch_size, num_layers, bl, ol)
        y_s = y_s.reshape(-1, self.num_layers, self.out_length)  # (batch_size, num_layers, ol)
        out=torch.sum(y_s,dim=1)
        return out  # (batch_size, out_length)

    # 移除所有数据分解方法
    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        """
        仅训练季节性模块
        """
        # 直接使用原始数据作为输入（不再分解趋势）
        x_train = x_train.to(self.device)
        x_val = x_val.to(self.device)
        y_train = y_train.to(self.device)
        y_val = y_val.to(self.device)

        # 仅训练季节性模块
        if self.args.train_seasonal_flag:
            self.seasonal.fit(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                batch_size=batch_size,
                epochs_sae=self.args.epochs_sae,
                epochs_kan=self.args.epochs_kan
            )
            self.seasonal.requires_grad_(False)
        self.seasonal.batch_size = batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        min_loss = float('inf')
        x_train=x_train[self.args.trend_length-self.args.input_length:]
        # 简化的训练循环
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for i in range(0, len(x_train) - batch_size + 1-self.args.input_length):
                optimizer.zero_grad()
                x_batch=x_train[i:i+batch_size+self.args.input_length-1]
                y_batch=[]
                for j in range(batch_size):
                    y_batch.append(y_train[i+j+self.args.input_length-self.args.out_length:i+j+self.args.input_length])
                y_batch=torch.stack(y_batch,dim=0).reshape(-1,self.args.out_length)
                y_pred = self(x_batch)
                loss = self.loss(y_pred, y_batch)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}')

    # 简化的测试方法
    def test(self, x_test, y_test_true=None, max=0.0, min=0.0):
        self.eval()
        x_test, _, _ = normalize(x_test)
        x_test=x_test[self.args.trend_length-self.args.input_length:]
        y_test_true=y_test_true[self.args.trend_length-self.args.input_length:]
        predictions = []
        for i in range(len(x_test) - self.args.input_length + 1):
            batch = x_test[i:i + self.args.input_length]
            pred = self.seasonal.test(batch)
            pred=torch.sum(pred,dim=0).reshape(-1, self.args.out_length)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0).to(self.device).reshape(-1, self.args.out_length)

        if y_test_true is not None:
            # 计算指标
            true = []
            for i in range(len(y_test_true) - self.args.input_length + 1):
                y_true_batch = y_test_true[i:i + self.args.input_length]
                true.append(y_true_batch[self.out_length * -1:, :])
            true = torch.stack(true, dim=0)
            y_true = true.reshape(-1, self.args.out_length)
            predictions = denormalize3(predictions, max, min)
            y_true = denormalize3(y_true, max, min)
            print(predictions.shape, y_true.shape)
            mse = torch.mean((predictions - y_true) ** 2)
            mask = y_true != 0
            mape = torch.mean(torch.abs((predictions[mask] - y_true[mask]) / y_true[mask])) * 100
            smape = torch.mean(2 * torch.abs(predictions - y_true) / (torch.abs(predictions) + torch.abs(y_true))) * 100
            print(f'Test MSE: {mse:.4f}, Test MAPE: {mape:.4f}%, Test SMAPE: {smape:.4f}%')
            # ploting(y_true.to('cpu'), predictions.to('cpu'))

        return predictions, y_true

    # 辅助方法
    def save_model(self, path):
        torch.save(self.state_dict(), f'model2/{path}.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))