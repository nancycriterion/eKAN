import itertools
import random

from data.read_data import ploting
from .data_V import data_div

from model.base_model.base_models import *

def STFSA(model,x_train,y_train,x_val,y_val,batch_size=10,length=24,T_max=33*37,S_max=50):

    param_grid={
        'batch_size':[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
        'length':[4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,
                  72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,
                  128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168]
    }
    all_combinations = list(itertools.product(param_grid['batch_size'], param_grid['length']))
    random.shuffle(all_combinations)

    a=0.0
    batch_size1,length1=batch_size,length
    model_temp=model
    model_temp.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, epoch=20, batch_size=batch_size1,
              length=length1)
    loss_val=model_temp.loss_val
    del model_temp
    t=0
    s=0
    while t<T_max:
        batch_size1,length1=all_combinations[t]
        model_temp=model
        model_temp.fit(x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,epoch=20,batch_size=batch_size1,
                  length=length1)
        loss_val1=model_temp.loss_val
        del model_temp
        if loss_val1+a*(batch_size1+length1)<loss_val+a*(batch_size+length):
            loss_val=loss_val1
            batch_size=batch_size1
            length=length1
            s=0
        else:
            s+=1
            if s>=S_max:
                break
        print(t,s,'个:', batch_size1, length1,'最佳：',batch_size,length,'loss=',loss_val)
        t+=1
    return batch_size,length

device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_cnn():
    x_train,y_train,x_val,y_val,x_test,y_test=data_div(3,3,True)
    model=CNNmodel(input_size=x_train.shape[1],hidden_size=8,output_size=y_train.shape[1],lr=0.001,device=device1)
    model.fit(x_train,y_train,x_val,y_val,epoch=100,batch_size=6,length=24)
    y_predict=model.predict(x_test)
    loss=model._loss(y_predict,y_test.to(device1))
    ploting(y_test[24:].to('cpu'), y_predict[:,24:].to('cpu'))
    print(loss.item())

def test_biGRU():
    x_train,y_train,x_val,y_val,x_test,y_test=data_div(3,3,True)
    model=BiGRUModel(input_size=x_train.shape[1],hidden_size=3,output_size=y_train.shape[1],device=device1,lr=0.01)
    model.fit(x_train,y_train,x_val,y_val,epoch=500,batch_size=6,length=48)
    y_predict=model.predict(x_test)
    loss=model._loss(y_predict,y_test.to(device1))
    ploting(y_test[24:].to('cpu'),y_predict[:,24:].to('cpu'))
    print(loss.item())

def test_cnn_gru():
    length=76
    x_train,y_train,x_val,y_val,x_test,y_test=data_div(3,1 ,True)
    model=CG(input_size=x_train.shape[1],hidden_size=8,output_size=y_train.shape[1],lr=0.001,device=device1)
    model.fit(x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,epoch=2,batch_size=6,length=length,flag=True)
    y_predict=model.predict(x_test)
    loss_mse=model._loss(y_predict,y_test.to(device1))
    loss_mspe = mspe_loss(y_predict, y_test.to(device1))
    ploting(y_test[length:].to('cpu'), y_predict[:,length:].to('cpu'))
    print(f'loss_mse:{loss_mse.item()}  loss_mspe:{loss_mspe.item()}%')

def mspe_loss(y_pred, y_true):
    # 避免分母为零的情况，加上一个小的常数 epsilon
    epsilon = 1e-30
    mask=y_true!=0
    y_pred=y_pred.reshape(-1,y_true.shape[1])
    percentage_errors = ((y_pred[mask] - y_true[mask]) / (y_true[mask] + epsilon)) ** 2
    mspe = torch.mean(percentage_errors) * 100
    return mspe
def  test_STFSA():
    x_train,y_train,x_val,y_val,x_test,y_test=data_div(3,3,True)
    model = CG(input_size=x_train.shape[1], hidden_size=8, output_size=y_train.shape[1], lr=0.001, device=device1)
    print("Before STFSA",flush=True)
    batch_size_set,length_set=STFSA(model,x_train,y_train,x_val,y_val,batch_size=6,length=24,S_max=25)
    print("After STFSA",batch_size_set,length_set,)
    model.fit(epoch=50,batch_size=batch_size_set,length=length_set,x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,flag=True)
    y_predict=model.predict(x_test)
    loss_mse=model._loss(y_predict,y_test.to(device1))
    loss_mspe=mspe_loss(y_predict,y_test.to(device1))
    ploting(y_test[length_set:].to('cpu'), y_predict[:,length_set:].to('cpu'))
    print(f'loss_mse:{loss_mse.item()}  loss_mspe:{loss_mspe.item()}%')