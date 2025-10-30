import numpy as np
import torch
from torch import nn
from data.read_data import get_data


def get_date_v(month):
    '''获取风速数据'''
    _, data = get_data(month)
    return data
def date_inout(x,y,step_size=1):
    '''用上一时间步的数据作为输入，预测下一时间步的风速'''
    l=len(x)
    x_in=x[:-step_size]
    y_out=y[step_size:]
    return x_in,y_out

def data_trend(x,y):
    '''L1滤波'''
    def trend_get(y):
        '''L1滤波'''
        y_tensor = torch.FloatTensor(y.values).unsqueeze(0)
        padding = (1, 1)  # 左右各 Padding 1
        y_padded = nn.functional.pad(y_tensor, padding,mode='replicate')
        # 平均池化操作
        avgpool = nn.AvgPool1d(kernel_size=3, stride=1)
        y_avg_pooled = avgpool(y_padded)
        # 将结果转换回 Pandas DataFrame
        y = y_avg_pooled.squeeze().detach().numpy()
        y=y.reshape(-1,1)
        return y
    y=trend_get(y)
    x=x.iloc[:,:4]
    # x=np.hstack((x,y))
    return x,y
def data_season(x,y):
    def trend_get(y):
        '''L1滤波'''
        y_tensor = torch.FloatTensor(y.values).unsqueeze(0)
        padding = (1, 1)  # 左右各 Padding 1
        y_padded = nn.functional.pad(y_tensor, padding,mode='replicate')
        # 平均池化操作
        avgpool = nn.AvgPool1d(kernel_size=3, stride=1)
        y_avg_pooled = avgpool(y_padded)
        # 将结果转换回 Pandas DataFrame
        y = y_avg_pooled.squeeze().detach().numpy()
        y=y.reshape(-1,1)
        return y
    y=y-trend_get(y)
    x=x.iloc[:,:4]
    return x.to_numpy(),y.to_numpy()
def data_div(month,step_size1=1,flag=False,flag_season=False):
    '''flag=True时，使用单变量时间预测
    step_size1: 单变量时间预测的步长'''
    x,y=get_data(month)
    if(flag_season==False):
        x,y=data_trend(x,y)
    else:
        x,y=data_season(x,y)
    if flag:
        x,y=date_inout(y,y,step_size1)
    else:
        x,y=date_inout(x,y,step_size1)
    train_bound=int(len(x)*0.6)
    val_bound=int(len(x)*0.8)
    x_train=x[:train_bound]
    y_train=y[:train_bound]
    x_val=x[train_bound:val_bound]
    y_val=y[train_bound:val_bound]
    x_test=x[val_bound:]
    y_test=y[val_bound:]
    x_train = torch.Tensor(x_train)
    x_val = torch.Tensor(x_val)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train)
    y_val = torch.Tensor(y_val)
    y_test = torch.Tensor(y_test)
    return x_train,y_train,x_val,y_val,x_test,y_test
