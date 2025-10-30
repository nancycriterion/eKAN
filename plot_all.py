import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data from the csv file
def read_data(file_name):
    data=pd.read_csv(file_name)
    return data.iloc[:,-1:]

def plot_all():
    file_name_list=[
                    # r'D:\code\zy\predict_use\models\data\result2\autoformer_ol_10_predict.csv',
                    r'D:\code\zy\predict_use\data\result2\fedformer_ol_1_predict.csv',
                    r'D:\code\zy\predict_use\data\result2\itransformer_ol_1_predict.csv',
                    r'D:\code\zy\predict_use\data\result2\non_stationary_transformer_ol_1_predict.csv',
                    r'D:\code\zy\predict_use\data\result2\timemixer_ol_1_predict.csv',
                    r'D:\code\zy\predict_use\data\result2\timenet_ol_1_predict.csv',
                    r'D:\code\zy\predict_use\data\result\stsakan3月_ol_1_predict.csv',
                    r'D:\code\zy\predict_use\data\result\stsakan3月_ol_1_true.csv',
                ]
    data_list=[]
    label_list=[
                # 'Autoformer',
                'Fedformer',
                'iTransformer',
                'NSTransformer',
                'TimeMixer',
                'TimeNet',
                'DMSAE-eKAN',
                'True'
            ]
    fmt_list=[['#c19a6b','--'],
              ['#9370db','--'],
              ['#4ecdc4','--'],
              ['#808080','--'],
              ['#ffaa71','--'],
              ['#ff6b6b','--'],
              ['#1f77b4','-']]
    plt.figure(figsize=(10, 3))
    for i,file_name in enumerate(file_name_list):
        data_list.append(read_data(file_name))
        plt.plot(data_list[i],label=label_list[i],color=fmt_list[i][0],linestyle=fmt_list[i][1],linewidth=1.5)
    plt.legend(loc='upper right',fontsize=10,bbox_to_anchor=(1, 1), ncol=4,frameon=True )
    plt.xlabel('Steps',fontsize=14)
    plt.ylabel('Value',fontsize=14)
    os.makedirs('./all',exist_ok=True)
    plt.savefig('./all/all_data.png', bbox_inches='tight', pad_inches=0.1,)
    plt.show()

    def mse(y_pred,y_true):
        return np.mean((y_true-y_pred)**2)
    def rmse(y_pred,y_true):
        return np.sqrt(mse(y_true,y_pred))
    def mape(y_pred,y_true):
        return np.mean(np.abs((y_true-y_pred)/y_true))
    def smape(y_pred,y_true):
        return 2*np.mean(np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)))
    mse_list=[]
    rmse_list=[]
    mape_list=[]
    smape_list=[]
    for i in range(len(data_list)):
        x=data_list[i].iloc[:,0].values
        y=data_list[-1].iloc[:,0].values
        mse_list.append(mse(x,y))
        rmse_list.append(rmse(x,y))
        mape_list.append(mape(x,y))
        smape_list.append(smape(x,y))
    print('label:',label_list)
    print('MSE:',mse_list)
    # print('RMSE:',rmse_list)
    print('MAPE:',mape_list)
    print('SMAPE:',smape_list)

def bar():
    data = [0.4729, 0.3174, 0.5027, 0.3277, 0.5686, 0.2497]
    label = ['Baseline0', 'Baseline1', 'Baseline2','Baseline3', 'Baseline4', 'Model']
    plt.figure(figsize=(10, 3))
    # 绘制柱状图，最后一个柱子使用不同颜色
    plt.bar(label, data, color=['#808080', '#808080', '#808080', '#808080', '#808080', '#ffaa71'])
    # plt.xlabel('Methods', fontsize=14)
    # plt.ylabel('MAPE', fontsize=14)
    plt.ylim(0, 0.6)
    # 将rotation改为0，使标签横向显示
    plt.xticks(rotation=0, ha='center', fontsize=16)  # ha='center'确保标签居中对齐
    os.makedirs('./bar', exist_ok=True)
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.show()

if __name__=='__main__':
    bar()