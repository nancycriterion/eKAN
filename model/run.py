import os
import pywt
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from .trend.STFSA import STFSA
from .base_model.base_models import CG
from .seasonal.seasonal import sae_kan_model
from data.read_data import ploting,normalize3,denormalize3,normalize
from .FOURIER.fourier import fourierTrendSeasonality


class FiLM(nn.Module):
    def __init__(self, num_layers, outlength):
        super(FiLM, self).__init__()
        self.num_layers = num_layers
        self.outlength = outlength

        # 1. 处理 condition: [batch, num_layers, outlength] -> [batch, outlength, 1]
        self.condition_proj = nn.Sequential(
            nn.Linear(num_layers, 1),  # 将 num_layers 维度压缩为 1
            nn.ReLU()
        )

        # 2. FiLM 变换: gamma * x + beta
        self.film = nn.Linear(1, 2)  # 生成 gamma 和 beta (每个 outlength 位置)

        # 3. 输出投影: [batch, outlength, 1] -> [batch, length]
        self.output_proj = nn.Linear(outlength, outlength)

    def forward(self, x, condition):
        """
        Args:
            x: [batch, outlength, 1]
            condition: [batch, num_layers, outlength]
        Returns:
            [batch, length]
        """
        batch_size = x.size(0)

        # Step 1: 处理 condition -> [batch, outlength, 1]
        condition = condition.permute(0, 2, 1)  # [batch, outlength, num_layers]
        condition = self.condition_proj(condition)  # [batch, outlength, 1]

        # Step 2: FiLM 变换
        gamma_beta = self.film(condition)  # [batch, outlength, 2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 [batch, outlength, 1]
        modulated = gamma * x + beta  # [batch, outlength, 1]

        # Step 3: 输出投影
        output = self.output_proj(modulated.squeeze(-1))  # [batch, length]
        return output


class stsakan(nn.Module):
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
        super(stsakan, self).__init__()
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
        self.seasonal = (sae_kan_model(input_size=input_size_s, input_length=args.input_length, out_length=out_length,
                                       num_layers=num_layers, lr=lr, device=device, args=args))
        self.film=FiLM(num_layers,out_length)
        self.linear=nn.Linear(num_layers+1,output_size)#可学习的输出门
        self.linear2=nn.Linear(num_layers,output_size)
        self.linear3=nn.Linear(num_layers+1,1)#reshape后线性变换在reshape回去
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
        # print(f'y_t.shape={y_t.shape}')
        y_s=y_s.permute(2,0,1,3)#seasonal处理的结果(num_layers,batch_length,batch_size,out_length)->(batch_size,num_layers,batch_length,out_length)
        y_s=y_s.reshape(-1,self.num_layers,self.out_length)#(b,nl,ol)  bl=1所以可以reshape
        # print(f't+s:y_t.shape={y_t.shape},y_s.shape={y_s.shape}')
        # y=torch.cat((y_t,y_s),dim=1)#(batch_size,num_layers+1,out_length)
        # y=self.film(y_t,y_s)#(batch_size,1,out_length)
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
            # print(f'y_s_sum.shape={y_s_sum.shape}',y_t.shape)
            out=y_t.reshape(-1,self.out_length)+y_s_sum
            out=self.relu(out)
        elif self.args.train_out_method=='FiLE':
            out=self.film(y_t,y_s)
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
        elif table=='wavelet':
            y_t,y_s=self.wavelet(y)
        else:
            y_t,y_s=y,y
        return y_t,y_s
    def fourier(self,y):
        y_t,y_s=self.fourier_model(y)
        return y_t,y_s
    def wavelet(self,y):#离散小波变换
        y_np=y.squeeze(1).cpu().numpy()
        cA, cD = pywt.dwt(y_np, 'db1')

        def pad_to_length(data, target_length):
            if len(data) < target_length:
                return np.pad(data, (0, target_length - len(data)), mode='constant')
            else:
                return data[:target_length]

        target_length = y.shape[0]
        cA = pad_to_length(cA, target_length)
        cD = pad_to_length(cD, target_length)

        # 转换回 PyTorch 张量并恢复形状
        cA_tensor = torch.from_numpy(cA).unsqueeze(1).to(y.dtype)
        cD_tensor = torch.from_numpy(cD).unsqueeze(1).to(y.dtype)

        return cA_tensor.to(self.device), cD_tensor.to(self.device)

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
        #保存xy_train_t和x_train到本地,用空格分隔
        # print(x_train.shape)
        np.savetxt(os.path.join('data','result','fourier')+'xy_train_t.txt',xy_train_t.to('cpu').detach().numpy(),fmt='%f',delimiter=' ')
        np.savetxt(os.path.join('data','result','fourier')+'x_train.txt',x_train.to('cpu').detach().numpy(),fmt='%f',delimiter=' ')
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
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            loss_s_p=0
            loss_t_p=0
            for i in range(0, len(x_train) - self.args.trend_length + 1 - batch_size):
                optimizer.zero_grad()
                xy_trend_batch = []
                y_batch = []
                y_train_s_batch=[]
                y_train_t_batch=[]
                for j in range(batch_size):
                    xy_trend_batch.append(xy_train_t[i + j:i + j + self.args.trend_length])
                    y_batch.append(y_train[i + j:i + j + self.args.trend_length])
                    y_train_s_batch.append(y_train_s[i + j:i + j + self.args.trend_length])
                    y_train_t_batch.append(y_train_t[i + j:i + j + self.args.trend_length])
                xy_trend_batch = torch.stack(xy_trend_batch, dim=0)
                y_batch = torch.stack(y_batch, dim=0)
                y_train_s_batch = torch.stack(y_train_s_batch, dim=0)
                # y_train_t_batch = torch.stack(y_train_t_batch, dim=0)
                x_train_batch = []
                x_train_batch.append(x_train[
                                     i + self.args.trend_length - self.input_length:i + batch_size + self.args.trend_length - 1])  # seasonal
                x_train_batch = torch.stack(x_train_batch, dim=0).reshape(-1, self.input_size_s)  # (b+il-1,i)
                y_seasonal = self.seasonal(x_train_batch)  # num_layers,batch_length,batch_size,out_length(bl=1)
                y_pred = self(xy_trend_batch, y_seasonal)  # (b,ol)
                y_batch = y_batch[:, self.args.out_length * -1:]  # (b,ol)
                y_train_s_batch = y_train_s_batch[:, self.args.out_length * -1:]  # (b,ol,1)
                # y_train_t_batch = y_train_t_batch[:, self.args.out_length * -1:]  # (b,ol,1)
                weight_s=self.weight[:,1:]
                # loss_t=self.loss(self.trend(xy_trend_batch)[:,-1*self.args.out_length:,:],y_train_t_batch.to(self.device))
                if self.args.train_out_method=='matrix':
                    loss_s = self.loss(
                        torch.einsum('on,nbl->lbo', weight_s, y_seasonal.reshape(self.num_layers, batch_size, -1)),
                        y_train_s_batch.permute(1, 0, 2).to(self.device))
                    loss = self.loss(y_pred.reshape(batch_size,-1), y_batch.reshape(batch_size, -1).to(self.device))\
                          # +loss_s*(1-1/self.args.fourier_order)+loss_t*(1/self.args.fourier_order)
                elif self.args.train_out_method=='gate':
                    loss=self.loss(y_pred.reshape(batch_size,-1), y_batch.reshape(batch_size, -1).to(self.device))
                elif self.args.train_out_method=='add' or self.args.train_out_method=='FiLE':
                    loss_s=self.loss(y_train_s_batch.permute(2,0,1).to(self.device),y_seasonal.sum(dim=0)/self.num_layers)
                    # loss = loss_s*(1-1/self.args.fourier_order)+loss_t*(1/self.args.fourier_order)
                    # loss_s_p+=loss_s.item()
                    # loss_t_p+=loss_t.item()
                    loss = self.loss(y_pred.reshape(batch_size, -1), y_batch.reshape(batch_size, -1).to(self.device))
                train_loss += loss
                loss.backward()
                optimizer.step()
            # print(f'loss_s_p={loss_s_p},loss_t_p={loss_t_p}')
            self.eval()
            with torch.no_grad():
                val_loss=0
                for i in range(0,len(x_val)-self.args.trend_length+1):
                    xy_trend_val_batch=[]
                    y_val_batch=[]
                    for j in range(1):
                        xy_trend_val_batch.append(xy_val_t[i+j:i+j+self.args.trend_length])
                        y_val_batch.append(y_val[i+j:i+j+self.args.trend_length])
                    xy_trend_val_batch=torch.stack(xy_trend_val_batch,dim=0)
                    y_val_batch=torch.stack(y_val_batch,dim=0)
                    x_val_batch=[]
                    x_val_batch.append(x_val[i+self.args.trend_length-self.input_length:i+1+self.args.trend_length-1])
                    x_val_batch=torch.stack(x_val_batch,dim=0).reshape(-1,self.input_size_s)
                    y_seasonal_val=self.seasonal.predict(x_val_batch)
                    y_pred_val=self(xy_trend_val_batch,y_seasonal_val)
                    y_val_batch=y_val_batch[:,self.args.out_length*-1:]
                    # y_val_batch=y_val_batch.reshape(-1,1)
                    val_loss+=self.loss(y_pred_val.reshape(1,-1),y_val_batch.reshape(1,-1).to(self.device))
                if val_loss<min_loss:
                    min_loss=val_loss
                    # self.save_model(f'best_model_{self.args.div_choose}{self.args.fourier_order}_ol_{self.args.out_length}_loss{min_loss}')
            if (epoch+1)%1==0:
                print(f'输出门训练：Epoch: {epoch+1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
    def test(self,x_test,y_test_true=None,max=1.0,min=0.0):
        self.eval()
        x_test,_,_=normalize(x_test)
        xy_test_div=x_test[:,-1:]
        x_test=x_test[:,:-1]
        xy_test_t,xy_test_s=self.div_choose(xy_test_div,table=self.args.div_choose)
        x_test=torch.cat((x_test,xy_test_s),dim=1)
        print(f'x_test.shape={x_test.shape},xy_test_t.shape={xy_test_t.shape}')
        out=[]
        for i in range(len(xy_test_t)-self.args.trend_length+1):
            xy_test_t_batch=xy_test_t[i:i+self.args.trend_length]
            xy_test_t_batch=xy_test_t_batch.unsqueeze(0)#(1,l,1)
            x_test_s=x_test[i + self.args.trend_length - self.input_length:i + self.args.trend_length ]
            y_test_s=self.seasonal.test(x_test_s)# num_layers,batch_length,batch_size,out_length(bl=1,bs=1)
            out.append(self(xy_test_t_batch,y_test_s))
        out=torch.stack(out,dim=0).reshape(-1,self.out_length)#(b,ol)
        predict=out
        true = []
        if y_test_true is not None:
            for i in range(len(y_test_true) - self.args.trend_length + 1):
                y_true_batch = y_test_true[i:i + self.args.trend_length]
                true.append(y_true_batch[self.out_length * -1:, :])
            true = torch.stack(true, dim=0)
            # y_test_true=y_test_true[self.input_length-1:,:]
            true = true.reshape(-1, self.args.out_length)
            # true_nor, max, min = normalize3(true)
            print(predict.shape, max,min)
            predict = denormalize3(predict, max,min)
            true = denormalize3(true, max,min)
            print(f'预测值：{predict.shape}')
            loss = self.loss(predict, true.to(self.device))
            print(f'预测值损失：{loss}')
            self.predict_loss(predict.to('cpu').detach().numpy(), true.to('cpu').detach().numpy())
            # print(f'预测值：{predict},实际值：{true}')
            # ploting(y_test_true.to('cpu'),predict.to('cpu').detach().numpy())
        return predict, true

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
            # true_nor,means,stds=normalize3(true)
            # predict=denormalize3(predict,means,stds)
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
        # y, x = predict.shape[0], predict.shape[1]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # for i in range(y):
        #     ax.plot(np.arange(x), np.full(x, i), predict[i], color='y', label='predict' if i == 0 else "")
        #     ax.plot(np.arange(x), np.full(x, i), y_test_true[i], color='r', label='true' if i == 0 else "")
        #
        # ax.set_ylabel('Time step (X)')
        # ax.set_xlabel('prediction length (Y)')
        # ax.set_zlabel('Value (Z)')
        # ax.set_xticks(np.arange(0, x, 1))  # 设置X轴刻度
        # ax.set_yticks(np.arange(0, y, 1))
        # ax.legend()
        # plt.show()
        predict=predict[:,-1:]
        y_test_true=y_test_true[:,-1:]
        ploting(y_test_true,predict)