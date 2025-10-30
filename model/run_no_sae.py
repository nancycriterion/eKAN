import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from .trend.STFSA import STFSA
from .base_model.base_models import CG
from .seasonal.seasonal import mul_kan_model
from data.read_data import ploting,normalize3,denormalize3,normalize
from .FOURIER.fourier import fourierTrendSeasonality

class stsakan_no_sae(nn.Module):
    def __init__(self, input_size_t, input_size_s, hidden_size, output_size, input_length, out_length, num_layers,
                 kernel_size, padding, lr, args=None, stride=1, device='cpu'):
        """

        :param input_size_t: trend输入维度
        :param input_size_s: seasonal输入维度
        :param hidden_size: trend隐藏层维度
        :param output_size: 1
        :param input_length: seasonal在sae中处理前的长度
        :param out_length: 1
        :param num_layers: seasonal里sae+kan的层数
        :param kernel_size: l1滤波池化层
        :param padding: l1滤波
        :param lr: 学习率
        :param stride: l1滤波
        :param device:
        """
        super(stsakan_no_sae, self).__init__()
        self.args=args
        self.input_size_t=input_size_t
        self.input_size_s=input_size_s
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.input_length=input_length
        self.out_length=out_length
        self.num_layers=num_layers
        self.lr=lr
        self.device=device
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding
            # 定义AvgPool1d层
        self.avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=self.padding)
        self.trend = (
            CG(input_size=input_size_t, hidden_size=hidden_size, output_size=output_size, lr=lr, device=device,
               args=args))
        self.seasonal=[]
        self.linear=[]
        for i in range(self.num_layers):
            self.seasonal.append(mul_kan_model(layers_hidden=[input_size_s, 6, 12, 6, 1], length=input_length,args=args, num_frequencies=10, scale_base=1.0, scale_fourier=1.0,
                 base_activation=nn.SiLU,device=device))
            self.linear.append(nn.Linear(input_length, out_length))
        self.seasonal=nn.ModuleList(self.seasonal)
        self.linear=nn.ModuleList(self.linear)
#可训练的权重矩阵
        self.weight=nn.Parameter(torch.rand(1,num_layers+1))
        self.bias=nn.Parameter(torch.rand(self.args.out_length))
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.gelu=nn.GELU()
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        self.loss=nn.MSELoss()
        self.fourier_model=fourierTrendSeasonality(args)
        self.to(device)

    def forward(self,y_t,y_s):
        '''
        :param x: 默认x最后一维是预测目标
        :return:
        '''
        # print(f't+s:y_t.shape={y_t.shape},y_s.shape={y_s.shape}')
        y_t=self.trend(y_t)#(batch_size,length,1)
        # y_t=self.gelu(y_t)
        y_t=y_t[:,-1*self.args.out_length:,:]#取最后一维，预测长度(batch_size,out_length,1)
        y_t=y_t.permute(0,2,1)#(batch_size,1,out_length)
        out_s=[]
        for i in range(self.num_layers):
            output=self.seasonal[i](y_s[i])
            out_s.append(self.linear[i](output.reshape(-1,self.input_length)))
        y_s=torch.stack(out_s,dim=1)
        y=torch.cat((y_t,y_s),dim=1)#(batch_size,num_layers+1,out_length)
        if self.args.train_out_method=='matrix':
            out=torch.einsum('on,bnl->bol',self.weight,y)
            out+=self.bias
            out=self.gelu(out)
        elif self.args.train_out_method=='gate':
            a=self.linear(y.reshape(-1,self.num_layers+1))
            a=(self.gelu(a)+1)/2
            out=a*y_t.reshape(-1,1)+(1-a)*y_s.reshape(-1,self.num_layers)#(b*ol,num_layers)
            out=self.linear2(out)#(b*ol,1)
            out=out.reshape(-1,self.out_length)#(b,ol)  # 将其形状调整为 (batch_size, out_length)
        elif self.args.train_out_method=='add':
            y_s_sum=torch.sum(y_s,dim=1)/self.num_layers#(batch_size,out_length)
            # y_s_sum=self.gelu(y_s_sum)
            out=y_t.reshape(-1,self.out_length)+y_s_sum
            out=self.gelu(out)
        return out


    def div_l1(self,y):
        #(length, 1)
        y_old=y
        y=y.unsqueeze(0)#( 1, length,1)
        y=y.permute(0,2,1)#(1, 1, length)
        # print('l1滤波',y.shape)
        y=self.avgpool(y)#(1, 1, length)
        y=y.squeeze(0).permute(1,0)#(length, 1)
        y_t=y
        y_s=y_old-y_t
        return y_t,y_s
    def div_choose(self,y,table='l1'):
        if table=='l1':
            y_t,y_s=self.div_l1(y)
        elif table=='fourier':
            y_t,y_s=self.fourier(y)
        else:
            y_t,y_s=y,y
        return y_t,y_s
    def fourier(self,y):
        y_t,y_s=self.fourier_model(y)
        return y_t,y_s

    def fit(self,x_train,y_train,x_val,y_val,epochs,batch_size):
        '''
        先分解时间序列，然后训练trend和seasonal，即forward
        输入：x_train(x|y)
        :return:
        '''
        # 训练trend和seasonal
        xy_train_div=x_train[:,-1:].to(self.device)#取最后一维，用于单变量trend训练
        xy_val_div=x_val[:,-1:].to(self.device)
        x_train=x_train[:,:-1].to(self.device)
        x_val=x_val[:,:-1].to(self.device)
        xy_train_t,xy_train_s=self.div_choose(xy_train_div,table=self.args.div_choose)#t用于训练trend，s用于训练seasonal
        xy_val_t,xy_val_s=self.div_choose(xy_val_div,table=self.args.div_choose)#验证集
        y_train_t,y_train_s=self.div_choose(y_train,table=self.args.div_choose)#训练集实际结果
        y_val_t,y_val_s=self.div_choose(y_val,table=self.args.div_choose)#验证集实际结果
        x_train=torch.cat((x_train,xy_train_s),dim=1)
        x_val=torch.cat((x_val,xy_val_s),dim=1)

        if self.args.plot_separate:
            ploting(y_train.to('cpu'),y_train_t.to('cpu').detach().numpy(),label='trend')
            ploting(y_train.to('cpu'),y_train_s.to('cpu').detach().numpy(),label='seasonal')
            ploting(y_train.to('cpu'),(y_train_t+y_train_s).to('cpu').detach().numpy())
        if self.args.train_trend_flag:
            self.trend.fit(x_train=xy_train_t, y_train=y_train_t, x_val=xy_val_t, y_val=y_val_t, batch_size=batch_size,
                           length=self.args.trend_length, epoch=self.args.epochs_CG)
            self.trend.requires_grad_(False)
        if self.args.train_seasonal_flag:
            self.seasonal.fit(x_train=x_train, y_train=y_train_s, x_val=x_val, y_val=y_val_s, batch_size=batch_size,
                              epochs_sae=self.args.epochs_sae, epochs_kan=self.args.epochs_kan)
            self.seasonal.requires_grad_(False)
        self.seasonal.batch_size=batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        min_loss=100
        x_train = self.fourier_model.separate(x_train)
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for i in range(0, x_train.shape[1] - self.args.trend_length + 1 - batch_size):
                optimizer.zero_grad()
                xy_trend_batch = []
                y_batch = []
                y_train_s_batch=[]
                y_train_t_batch=[]
                x_train_batch = []
                for j in range(batch_size):
                    xy_trend_batch.append(xy_train_t[i + j:i + j + self.args.trend_length])
                    y_batch.append(y_train[i + j:i + j + self.args.trend_length])
                    y_train_s_batch.append(y_train_s[i + j:i + j + self.args.trend_length])
                    y_train_t_batch.append(y_train_t[i + j:i + j + self.args.trend_length])
                    x_train_batch.append(x_train[:,i+j+self.args.trend_length - self.input_length:i+j+self.args.trend_length])
                xy_trend_batch = torch.stack(xy_trend_batch, dim=0)
                y_batch = torch.stack(y_batch, dim=0)
                x_train_batch=torch.stack(x_train_batch,dim=1)#季节性输入(b,il,n)
                # print(xy_trend_batch.shape,x_train_batch.shape)
                y_pred = self(xy_trend_batch, x_train_batch)  # (b,ol)
                y_batch = y_batch[:, self.args.out_length * -1:]  # (b,ol)
                loss = self.loss(y_pred.reshape(batch_size, -1), y_batch.reshape(batch_size, -1).to(self.device))
                train_loss += loss
                loss.backward()
                optimizer.step()
            if (epoch+1)%1==0:
                print(f'输出门训练：Epoch: {epoch+1}, Train Loss: {train_loss:.5f}')
    def test(self,x_test,y_test_true=None,max=0.0,min=0.0):
        self.eval()
        x_test,_,_=normalize(x_test)
        xy_test_div=x_test[:,-1:]
        x_test=x_test[:,:-1]
        xy_test_t,xy_test_s=self.div_choose(xy_test_div,table=self.args.div_choose)
        x_test=torch.cat((x_test,xy_test_s),dim=1)
        x_test=self.fourier_model.separate(x_test)
        out=[]
        true=[]
        for i in range(x_test.shape[1]-self.args.trend_length+1):
            xy_test_t_batch=xy_test_t[i:i+self.args.trend_length]
            xy_test_t_batch=xy_test_t_batch.unsqueeze(0)#(1,l,1)
            x_test_s=x_test[:,i+self.args.trend_length-self.input_length:i+self.args.trend_length]
            out.append(self(xy_test_t_batch,x_test_s))
            true.append(y_test_true[i+self.args.trend_length-self.out_length:i+self.args.trend_length])
        predict=torch.stack(out,dim=0).reshape(-1,self.args.out_length)#(b,ol)
        true=torch.stack(true,dim=0).reshape(-1,self.args.out_length)#(b,ol)
        predictions = denormalize3(predict, max, min)
        y_true = denormalize3(true, max, min)
        mse = torch.mean((predictions - y_true) ** 2)
        mape = torch.mean(torch.abs((predictions - y_true) / y_true)) * 100
        smape = torch.mean(2 * torch.abs(predictions - y_true) / (torch.abs(predictions) + torch.abs(y_true))) * 100
        print(f'Test MSE: {mse:.4f}, Test MAPE: {mape:.4f}%, Test SMAPE: {smape:.4f}%')
        return predictions, y_true

    def predict(self,x_test,y_test_true=None):
        self.eval()
        x_test,_,_=normalize(x_test)
        xy_test_div=x_test[:,-1:]
        x_test=x_test[:,:-1]
        x_test=x_test[self.args.trend_length-self.input_length:]
        xy_test_t,_=self.div_choose(xy_test_div,table=self.args.div_choose)
        print(f'x_test.shape={x_test.shape},xy_test_t.shape={xy_test_t.shape}')
        y_test_s=self.seasonal.predict(x_test)#(nl,bl,b,ol):b=1,bl=l-il+1
        y_test_s=y_test_s.permute(2,0,1,3)#(b,nl,bl,ol)
        y_test_s=y_test_s.reshape(-1,self.num_layers,self.out_length)#(bl,nl,ol)
        # y_test_s=y_test_s.reshape(-1,self.num_layers)
        y_trend=[]
        for i in range(len(xy_test_t)-self.args.trend_length+1):
            xy_test_t_batch=xy_test_t[i:i+self.args.trend_length]
            xy_test_t_batch=xy_test_t_batch.unsqueeze(0)#(1,l,1)
            y_test_t_batch=self.trend(xy_test_t_batch).reshape(1,-1)#(1,l)
            y_test_t_batch=self.sigmoid(y_test_t_batch)
            y_trend.append(y_test_t_batch[:,self.out_length*-1:])
        y_trend=torch.stack(y_trend,dim=0)#(bl,1,ol)
        # print(f'预测trend：{y_trend.shape},预测seasonal：{y_test_s.shape}')
        y_test=torch.cat((y_trend,y_test_s),dim=1)#(bl,nl+1,ol)
        # y_test=y_test.reshape(-1,self.num_layers+1)#(bl*ol,nl+1)
        # out=self.linear3(y_test)#(bl*ol,1)
        out = torch.einsum('on,bnl->bol', self.weight, y_test)
        out += self.bias
        out = self.sigmoid(out)
        # a=self.linear(y_test)
        # a=(self.tanh(a)+1)/2
        # out=a*y_trend.reshape(-1,1)+(1-a)*y_test_s.reshape(-1,self.num_layers)
        # out=self.linear2(out)#(bl*ol,1)
        # out=self.sigmoid(out)
        out=out.reshape(-1,self.out_length)#(bl,ol)
        predict=out
        true = []
        if y_test_true is not None:
            for i in range(len(y_test_true)-self.args.trend_length+1):
                y_true_batch=y_test_true[i:i+self.args.trend_length]
                true.append(y_true_batch[self.out_length*-1:,:])
            true=torch.stack(true,dim=0)
            # y_test_true=y_test_true[self.input_length-1:,:]
            true=true.reshape(-1,self.args.out_length)
            true_nor,means,stds=normalize3(true)
            predict=denormalize3(predict,means,stds)
            print(f'预测值：{predict.shape}')
            loss=self.loss(predict,true.to(self.device))
            print(f'预测值损失：{loss}')
            self.predict_loss(predict.to('cpu').detach().numpy(),true.to('cpu').detach().numpy())
            # print(f'预测值：{predict},实际值：{true}')
            # ploting(y_test_true.to('cpu'),predict.to('cpu').detach().numpy())
        return predict,true
    def predict_loss(self,predict,y_test_true):
        mse=np.mean((predict-y_test_true)**2)
        mae=np.mean(np.abs(predict-y_test_true))
        rmse=np.sqrt(mse)
        mask = y_test_true != 0
        mape = np.mean(np.abs((predict[mask] - y_test_true[mask]) / y_test_true[mask])) * 100
        smape=np.mean(np.abs(predict-y_test_true)/(np.abs(predict)+np.abs(y_test_true)))*2*100
        nrmse=rmse/np.std(y_test_true)*100
        print(f'mse:{mse},mae:{mae},rmse:{rmse},mape:{mape:.4f}%,smape:{smape:.4f}%,nrmse:{nrmse:.4f}%')
        self.predict_3D(predict,y_test_true)
    def save_model(self,path):
        torch.save(self,'model2/'+path+'.pth')
    def load_model(self,path):
        self.load_state_dict(torch.load(path),strict=False)
    def predict_3D(self,predict,y_test_true):
        y, x = predict.shape[0], predict.shape[1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(y):
            ax.plot(np.arange(x), np.full(x, i), predict[i], color='y', label='predict' if i == 0 else "")
            ax.plot(np.arange(x), np.full(x, i), y_test_true[i], color='r', label='true' if i == 0 else "")

        ax.set_ylabel('Time step (X)')
        ax.set_xlabel('prediction length (Y)')
        ax.set_zlabel('Value (Z)')
        ax.set_xticks(np.arange(0, x, 1))  # 设置X轴刻度
        ax.set_yticks(np.arange(0, y, 1))
        ax.legend()
        plt.show()
        predict=predict[:,-1:]
        y_test_true=y_test_true[:,-1:]
        ploting(y_test_true,predict)