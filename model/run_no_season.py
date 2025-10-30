import numpy as np
import torch
from torch import nn
from .trend.STFSA import STFSA
from .base_model.base_models import CG
from data.read_data import ploting, normalize3, denormalize3, normalize
from .FOURIER.fourier import fourierTrendSeasonality


class stsakan_no_season(nn.Module):  # 修改类名以区分消融版本
    def __init__(self, input_size_t, input_size_s, hidden_size, output_size, input_length, out_length, num_layers,
                     kernel_size, padding, lr, args=None, stride=1, device='cpu'):
        """
        消融版本：移除seasonal模块后的模型
        """
        super(stsakan_no_season, self).__init__()
        self.args = args
        self.input_size_t = input_size_t
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_length = input_length
        self.out_length = out_length
        self.lr = lr
        self.device = device

        # 移除了所有seasonal相关参数
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding

        # 仅保留趋势相关模块
        self.avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=self.padding)
        self.trend = CG(
            input_size=input_size_t,
            hidden_size=hidden_size,
            output_size=output_size,
            lr=lr,
            device=device,
            args=args
        )
        self.stfsa = STFSA
        self.loss = nn.MSELoss()
        self.fourier_model = fourierTrendSeasonality(args)
        self.to(device)

    def forward(self, y_t):
        """
        简化后的前向传播：仅使用趋势模块
        """
        y_t = self.trend(y_t)  # (batch_size, length, 1)
        y_t = y_t[:, -self.args.out_length:, :]  # 取预测长度
        return y_t.squeeze(-1)  # 输出形状 (batch_size, out_length)

    # 保留数据分解方法（但仅使用趋势部分）
    def div_l1(self, y):
        y_old = y
        y = y.unsqueeze(0).permute(0, 2, 1)
        y = self.avgpool(y)
        y_t = y.squeeze(0).permute(1, 0)
        return y_t, torch.zeros_like(y_t)  # 季节部分置零

    def div_choose(self, y, table='l1'):
        if table == 'l1':
            y_t, _ = self.div_l1(y)  # 忽略季节部分
        elif table == 'fourier':
            y_t, _ = self.fourier(y)
        else:
            y_t = y
        return y_t, None  # 季节部分返回None
    def fourier(self,y):
        y_t,y_s=self.fourier_model(y)
        return y_t,y_s

    # 简化的训练方法
    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        """
        仅训练趋势模块
        """
        xy_train_div = x_train[:, -1:].to(self.device)
        xy_val_div = x_val[:, -1:].to(self.device)

        # no数据分解
        xy_train_t=xy_train_div
        xy_val_t=xy_val_div
        y_train_t=y_train
        y_val_t=y_val

        # 仅训练趋势模块
        if self.args.train_trend_flag:
            self.trend.fit(
                x_train=xy_train_t,
                y_train=y_train_t,
                x_val=xy_val_t,
                y_val=y_val_t,
                batch_size=batch_size,
                length=self.args.trend_length,
                epoch=self.args.epochs_CG
            )
            self.trend.requires_grad_(False)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        min_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for i in range(0, len(xy_train_t) - self.args.trend_length + 1 - batch_size):
                optimizer.zero_grad()
                xy_trend_batch_all=xy_train_t[i:i+batch_size+self.args.trend_length-1]
                xy_trend_batch=[]
                y_batch_all=y_train_t[i:i+batch_size+self.args.trend_length-1]
                y_batch=[]
                for j in range(batch_size):
                    xy_trend_batch.append(xy_trend_batch_all[j:j+self.args.trend_length])
                    y_batch.append(y_batch_all[j+self.args.trend_length-self.out_length:j+self.args.trend_length])
                xy_trend_batch=torch.stack(xy_trend_batch,dim=0).to(self.device)
                y_batch=torch.stack(y_batch,dim=0).to(self.device).reshape(batch_size,-1)
                # print(xy_trend_batch_all.shape,xy_trend_batch.shape)
                # 仅处理趋势数据

                y_pred = self(xy_trend_batch)
                # print(y_pred.shape,y_batch.shape)
                loss = self.loss(y_pred, y_batch.to(self.device))
                train_loss += loss.item()


                loss.backward()
                optimizer.step()

            # 验证步骤
            self.eval()
            with torch.no_grad():
                val_loss = 0
                for i in range(0, len(xy_val_t) - self.args.trend_length + 1):
                    xy_trend_val = xy_val_t[i:i + self.args.trend_length].unsqueeze(0)
                    y_pred_val = self(xy_trend_val)
                    val_loss += self.loss(y_pred_val, y_val_t[i:i + 1].to(self.device)).item()

                if val_loss < min_loss:
                    min_loss = val_loss
                    self.save_model(f'ablated_best_{min_loss:.4f}')

            print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}')

    # 简化的测试方法
    def test(self, x_test, y_test_true=None, max=0.0, min=0.0):
        self.eval()
        x_test, _, _ = normalize(x_test)
        xy_test_div = x_test[:, -1:]
        xy_test_t, _ = self.div_choose(xy_test_div, table=self.args.div_choose)

        predictions = []
        for i in range(len(xy_test_t) - self.args.trend_length + 1):
            batch = xy_test_t[i:i + self.args.trend_length].unsqueeze(0)
            pred = self(batch)
            predictions.append(pred)

        predictions=torch.stack(predictions,dim=0).to(self.device).reshape(-1,self.args.out_length)

        if y_test_true is not None:
            # 计算指标
            true = []
            for i in range(len(y_test_true) - self.args.trend_length + 1):
                y_true_batch = y_test_true[i:i + self.args.trend_length]
                true.append(y_true_batch[self.out_length * -1:, :])
            true = torch.stack(true, dim=0)
            y_true = true.reshape(-1, self.args.out_length)
            predictions = denormalize3(predictions, max, min)
            y_true = denormalize3(y_true, max, min)
            mse = torch.mean((predictions - y_true) ** 2)
            mape = torch.mean(torch.abs((predictions - y_true) / y_true)) * 100
            smape= torch.mean(2 * torch.abs(predictions - y_true) / (torch.abs(predictions) + torch.abs(y_true))) * 100
            print(f'Test MSE: {mse:.4f}, Test MAPE: {mape:.4f}%, Test SMAPE: {smape:.4f}%')
            # ploting(y_true.to('cpu'), predictions.to('cpu'))

        return predictions,y_true

    # 保留的辅助方法
    def save_model(self, path):
        torch.save(self.state_dict(), f'model2/{path}.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))