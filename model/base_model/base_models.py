import numpy as np
import torch
import torch.nn as nn
from torch import optim


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding - 1,
                               dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation)
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(), self.dropout1, self.conv2, nn.ReLU(), self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.size(-1) > res.size(-1):
            out = out[:, :, :res.size(-1)]  # 裁剪 out 的时间维度
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_length=5):
        super(TCN, self).__init__()
        layers = []
        num_layers = len(num_channels)
        # print('n=',num_layers)
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_length)

    def forward(self, x):
        # print('tcn.shape1=', x.shape)
        x = self.network(x)
        # print('tcn.shape2=', x.shape)
        x = x.mean(dim=-1)
        # print('tcn.shape3=', x.shape)
        x = self.fc(x)
        # print('tcn.shape4=',x.shape)

        return x

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMmodel, self).__init__()
        self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc=nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out=lstm_out[:,-1,:]
        out=self.fc(lstm_out)
        return out
    def fit(self,x_train,y_train,x_test,y_test,epochs,batch_size,lr,device='cpu'):
        self.to(device)
        criterion=nn.MSELoss()
        optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        for epoch in range(epochs):
            loss=0
            for i in range(0,x_train.shape[0],batch_size):
                x_batch=x_train[i:i+batch_size].to(device)
                y_batch=y_train[i:i+batch_size].to(device)
                optimizer.zero_grad()
                y_pred=self(x_batch)
                loss=criterion(y_pred,y_batch)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                y_pred_val=self(x_test.to(device))
                loss_val=criterion(y_pred_val,y_test.to(device))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {loss_val.item():.4f}")

    def predict(self,x_test,device):
        self.to(device)
        with torch.no_grad():
            y_pred_test=self(x_test.to(device))
        return y_pred_test.cpu().numpy()

class CNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1,device='cpu',lr=0.001):
        super(CNNmodel, self).__init__()
        self.conv1=nn.Conv1d(input_size,hidden_size,kernel_size=3,padding=1)
        self.pool1=nn.MaxPool1d(kernel_size=2,stride=2)
        self.conv2=nn.Conv1d(int(hidden_size/2),hidden_size,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool1d(kernel_size=2,stride=2)

        self.full = nn.Linear(int(hidden_size/2), output_size)
        self.device=device
        self._loss = nn.MSELoss()
        self.lr=lr
        self.length=0

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.conv1(x)
        x = x.permute(0, 2, 1)
        x=self.pool1(x)
        x = x.permute(0, 2, 1)
        x=self.conv2(x)
        x = x.permute(0, 2, 1)
        x=self.pool2(x)
        x=self.full(x)
        #print(x.shape)
        return x
    def fit(self,x_train,y_train,x_val,y_val,epoch,batch_size,length,lr=0.001):
        self.length=length
        lr=self.lr
        device=self.device
        self.to(device)
        criterion=nn.MSELoss()
        loss=0
        optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        for i in range(epoch):
            self.train()

            for j in range(0,x_train.shape[0]-length+1-batch_size,batch_size):
                x_batch=[]
                y_batch=[]
                optimizer.zero_grad()
                for k in range (batch_size):
                    x_batch.append(x_train[j+k:j+k+length])
                    y_batch.append(y_train[j+k:j+k+length])
                x_batch=torch.stack(x_batch,dim=0).to(device)#[batch_size,length,input_size]
                y_batch=torch.stack(y_batch,dim=0).to(device)#[batch_size,length,output_size]
                # print(x_batch.shape,y_batch.shape)
                y_pred=self(x_batch)
                loss=criterion(y_pred,y_batch)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                x_batch_val=[]
                y_batch_val=[]
                for j in range(0,x_val.shape[0]-length-1):
                    x_batch_val.append(x_val[j:j+length])
                    y_batch_val.append(y_val[j:j+length])
                x_batch_val=torch.stack(x_batch_val,dim=0).to(device)
                y_batch_val=torch.stack(y_batch_val,dim=0).to(device)
                # print(x_batch_val.shape,y_batch_val.shape)
                y_pred_val=self.predict(x_batch_val.to(device))
                # print(y_pred_val.dtype,y_val.dtype)
                loss_val=criterion(y_pred_val,y_batch_val.to(device))
                print(f"Epoch {i+1}/{epoch}, Loss: {loss.item():.4f}, Val Loss: {loss_val.item():.4f}")

    def predict(self,x_test):
        device = self.device
        self.to(device)
        if len(x_test.shape) == 2:
            x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
        with torch.no_grad():
            x_batch = x_test[:, 0:self.length, :]
            y_pred_test = self(x_batch.to(device))
            for i in range(1, x_test.shape[1] - self.length + 1):
                x_batch = x_test[:, i:i + self.length, :]
                y_pred = self(x_batch.to(device))
                y_pred_test = torch.cat((y_pred_test, y_pred[:, self.length - 1:]), dim=1)
        return y_pred_test

import torch.nn.functional as F

class DilatedCNN(nn.Module):
    def __init__(self,input_length):
        super(DilatedCNN, self).__init__()
        self.length=input_length
        # 定义带有不同空洞率的卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                               dilation=1, padding=1)  # 标准卷积
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               dilation=2, padding=2)  # 空洞率2
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               dilation=4, padding=4)  # 空洞率4

        # 其他层
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc = nn.Linear(32 * int(input_length),input_length)  # 假设输入是224x224

    def forward(self, x):

        x = F.relu(self.conv1(x))
        # print('1',x.shape)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print('2',x.shape)
        x = F.relu(self.conv3(x))
        # print('3',x.shape)
        x = self.pool(x)
        # print('4',x.shape)
        x = x.view(-1, 32 * int(self.length))
        x = self.fc(x)
        return x

class BiGRUModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, output_size=1,device='cpu',lr=0.001):
        super(BiGRUModel, self).__init__()
        self.device=device
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 双向 GRU 的输出维度是 hidden_size * 2
        self._loss = nn.MSELoss()
        self.to(device)
        self.lr=lr
        self.length=0


    def forward(self, x):
        # x 的形状是batch_size,seq_len,input_size)
        # print('gru:',x.shape)
        out, _ = self.gru(x)  # out 的形状是 (batch_size, seq_len, hidden_size * 2)
        # print(out.shape)
        # 将 GRU 的输出传入全连接层
        out = self.fc(out)  # out 的形状是 (batch_size, seq_len, output_size)
        return out
    def fit(self,x_train,y_train,x_val,y_val,epoch,batch_size,length,lr=0.001):
        self.length=length
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        for i in range(epoch):
            self.train()

            loss=0
            for j in range(0,x_train.shape[0]-length+1-batch_size,batch_size):
                x_batch=[]
                y_batch=[]
                optimizer.zero_grad()
                for k in range(batch_size):
                    x_batch.append(x_train[j+k:j+k+length])
                    y_batch.append(y_train[j+k:j+k+length])
                x_batch=torch.stack(x_batch,dim=0).to(self.device)
                y_batch=torch.stack(y_batch,dim=0).to(self.device)
                # print(x_batch.shape,y_batch.shape)
                y_pred=self(x_batch)
                loss=self._loss(y_pred,y_batch)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                y_pred_val=self.predict(x_val.to(self.device))
                loss_val=self._loss(y_pred_val,y_val.to(self.device))
                print(f"Epoch {i+1}/{epoch}, Loss: {loss.item():.4f}, Val Loss: {loss_val.item():.4f}")

    def predict(self,x_test):
        device = self.device
        self.to(device)
        if len(x_test.shape) == 2:
            x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
        with torch.no_grad():
            x_batch = x_test[:, 0:self.length, :]
            y_pred_test = self(x_batch.to(device))
            for i in range(1, x_test.shape[1] - self.length + 1):
                x_batch = x_test[:, i:i + self.length, :]
                y_pred = self(x_batch.to(device))
                y_pred_test = torch.cat((y_pred_test, y_pred[:, self.length - 1:]), dim=1)
        return y_pred_test


def mspe_loss(y_pred, y_true):
    # 避免分母为零的情况，加上一个小的常数 epsilon
    epsilon = 1e-10
    percentage_errors = ((y_pred - y_true) / (y_true + epsilon)) ** 2
    mspe = torch.mean(percentage_errors) * 100
    return mspe


class TimeAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TimeAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # print('TimeAttention.shape1=',x.shape)
        attention_weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        weighted_output = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_size)
        return weighted_output, attention_weights


class CG(nn.Module):
    def __init__(self,input_size,hidden_size,args=None,output_size=1,lr=0.01,device='cpu'):
        super(CG, self).__init__()
        self.device=device
        self.args=args
        # self.cnn =CNNmodel(input_size=input_size,hidden_size=hidden_size,output_size=output_size,lr=0.001,device=device)
        self.cnn=DilatedCNN(args.trend_length)
        self.sigmoid = nn.Sigmoid()
        self.bigru =BiGRUModel(input_size=args.trend_length,hidden_size=hidden_size,output_size=output_size,lr=0.01,device=device)
        self.time_attention = TimeAttention(hidden_size)
        self._opt = optim.Adam(self.parameters(), lr=lr)
        self._loss = nn.MSELoss()
        self.loss_val=0
        self.length=0
        self.linear = nn.Linear(hidden_size, 1)  # 修改输入尺寸为hidden_size
        self.linear2=nn.Linear(32,output_size)
        self.gelu=nn.GELU()
        self.to(device)
        self.output_size=output_size
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x):
        # print('x.shape1=',x.shape)
        x = x.unsqueeze(1)  # 增加时间维度 (batch_size, 1, input_size)
        # print('x.shape2=',x.shape)
        x = self.cnn(x)
        x=self.dropout(x)
        # print('cnn.shape1=',x.shape)
        x = self.sigmoid(x)
        # print('sigmoid.shape2=',x.shape)
        x = self.bigru(x.unsqueeze(1))
        # print('bigru.shape3=',x.shape)
        x, attention_weights = self.time_attention(x)  # x shape: (batch_size, hidden_size)
        x = x.unsqueeze(1)  # 恢复时间维度 (batch_size, 1, hidden_size)
        x=self.sigmoid(x)
        # print('gru.shape3=',x.shape,self.output_size)
        x = self.linear2(x)
        # print('linear2.shape4=',x.shape)
        return x
    def fit(self,x_train,y_train,x_val,y_val,epoch=10,batch_size=8,length=24,flag=True):
        self.length=length
        for i in range(epoch):
            loss_train=0
            loss_val=0
            self.train()
            for j in range(0,len(x_train)-length+1-batch_size,batch_size):
                x_batch=[]
                y_batch=[]
                self._opt.zero_grad()
                for k in range(batch_size):
                    x_batch.append(x_train[j+k:j+k+length])
                    y_batch.append(y_train[j+k:j+k+length])
                x_batch=torch.stack(x_batch,dim=0).to(self.device)
                y_batch=torch.stack(y_batch,dim=0).to(self.device)
                # print(x_batch.shape,y_batch.shape)
                x=x_batch
                y=y_batch
                output = self(x)
                # print('CG.shape1=',output.shape,y.shape)
                loss = self._loss(output[:,self.args.out_length*-1:,:],y[:,self.args.out_length*-1:,:])
                loss.backward()
                loss_train+=loss.item()
                self._opt.step()
            self.eval()
            with torch.no_grad():
                y_pred_val=self.predict(x_val).reshape(-1,1)
                loss_val=self._loss(y_pred_val[-self.args.out_length:],y_val[-self.args.out_length:].to(self.device))
            self.loss_val=loss.item()
            if flag and (i+1)%10==0:
                print('CG::::Epoch:{},loss:{:.4f},val_loss:{:.4f}'.format(i+1,loss_train,loss_val.item()))

    def predict(self,x):
        device = self.device
        self.to(device)
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        with torch.no_grad():
            x_batch = x[:,0:self.length,:]
            y_pred_test=self(x_batch.to(device))
            for i in range(1,x.shape[1]-self.length+1):
                x_batch = x[:,i:i+self.length,:]
                y_pred = self(x_batch.to(device))
                y_pred_test=torch.cat((y_pred_test,y_pred[:,self.length-1:]),dim=1)
        if self.args.out_length>1:#此函数在总模型中没有使用
            pass
        return y_pred_test
