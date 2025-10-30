import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


def get_data(month,normalize_flag=False):
    '''
    data_x:气压    气温  相对湿度  10分钟平均能见度 10分钟平均风速
    data_y:10分钟平均风速
    '''
    file="D:\\code\\zy\\predict\\data\\20230308\\"+str(month)+"月.xlsx"
    data=read_file(file)
    data_x,data_y=div_data(data)
    if normalize_flag:
        data_x,data_y=normalize_1(data_x),normalize_1(data_y)
    return data_x,data_y

def train_test_split1(x, y, test_size=0.2):
    l=len(x)
    t=l-int(l*test_size)
    x_train=x[:t]
    x_test=x[t:]
    y_train=y[:t]
    y_test=y[t:]
    return x_train,x_test,y_train,y_test

def read_file(file):
    data=pd.read_excel(file)
    return data

def div_data(data):
    data_x=pd.concat([data.iloc[:,2:5],data.iloc[:,7:8],data.iloc[:,6:7]],axis=1)
    data_y=pd.concat([data.iloc[:,6:7]],axis=1)
    # data_x=data_y
    return data_x,data_y

def normalize_1(data):
    scalar=MinMaxScaler()
    normalize_data=pd.DataFrame(scalar.fit_transform(data),columns=data.columns)
    return normalize_data

def normalize2(y):
    means=y.mean(0,keepdim=True).detach()
    # print(means.shape,y.shape)
    y=y-means
    stds=torch.sqrt(torch.var(y,dim=0,keepdim=True,unbiased=False)+1e-6)
    y=y/stds
    return y,means,stds
def denormalize2(y,means,stds):
    y=y*stds
    y=y+means
    return y
def normalize(data):
    #取每一列的最大最小值
    max=data.max(axis=0)
    min=data.min(axis=0)
    data=(data-min.values)/(max.values-min.values)
    # print(max,min)
    return data,max,min
def denormalize(data,max,min):
    data=data*(max.values-min.values)+min.values
    return data
def normalize3(data):
    max1=data.max()
    min1=data.min()
    data=(data-min1)/(max1-min1)
    # print(max,min)
    return data,max1,min1
def denormalize3(data,max,min):
    data=data*(max-min)+min
    return data
def ploting(y_test,y_pred,label=None):
    import matplotlib.pyplot as plt
    y_test=y_test.reshape(-1,1)
    y_pred=y_pred.reshape(-1,1)
    if label is None:
        plt.plot(y_test,label="y_test")
        #线条为虚线
        plt.plot(y_pred,label="y_pred",linestyle="--")
    elif label =="trend":
        plt.plot(y_test,label="y_true")
        plt.plot(y_pred,label="y_trend",linestyle="--")
    elif label =="seasonal":
        plt.plot(y_test,label="y_true")
        plt.plot(y_pred,label="y_seasonal", linestyle="--")
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    mse = np.mean((y_test - y_pred) ** 2)#均方误差
    rmse = np.sqrt(mse)#均方根误差
    mae = np.mean(np.abs(y_test - y_pred))#平均绝对误差
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100#平均绝对百分比误差
    smape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))) * 100#改良的平均绝对百分比误差

    # 计算 R²
    # r_squared = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")
    # print("R²:", r_squared)

def ploting_1(x):
    import matplotlib.pyplot as plt
    plt.plot(x,label='x')
    plt.legend()
    plt.show()
def read_csv(str):
    file='D:\\code\\zy\\predict_241127\\data\\20230308\\'+str
    data=pd.read_csv(file)
    data_x,data_y=pd.concat([data.iloc[:,2:6],data.iloc[:,1:2]],axis=1),data.iloc[:,1:2]
    return data_x[9000:12000],data_y[9000:12000]
def read_csv1(str):
    script_dir=os.path.dirname(__file__)
    file_path=os.path.dirname(script_dir)
    file=os.path.join(file_path,'1','all_datasets','all_datasets',str)
    data=pd.read_csv(file)
    data_x,data_y=pd.concat([data.iloc[:,2:6],data.iloc[:,1:2]],axis=1),data.iloc[:,1:2]
    return data_x[9000:12000],data_y[9000:12000]


if __name__=="__main__":
    read_csv1('20230308.csv')
    # print(get_data(3))