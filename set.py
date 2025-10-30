import os

import numpy as np
import torch
import time
from model.run import stsakan
from data.read_data import *
import argparse
from model import *

parser = argparse.ArgumentParser(description='Train STS-AKAN model')
parser.add_argument('--month',type=int,default=3,help='month of data')
parser.add_argument('--normalize_flag',type=bool,default=False,help='normalize flag')
parser.add_argument('--train_size',type=float,default=0.7,help='train size')
parser.add_argument('--val_size',type=float,default=0.15,help='val size')
# parser.add_argument('--step_size',type=int,default=1,help='step size')
parser.add_argument('--hidden_size',type=int,default=32,help='hidden size')
parser.add_argument('--num_layers',type=int,default=3,help='num layers5')
parser.add_argument('--kernel_size',type=int,default=3,help='kernel size')
parser.add_argument('--padding',type=int,default=32,help='padding')
parser.add_argument('--out_length',type=int,default=48,help='out length')
parser.add_argument('--epochs_sae',type=int,default=100,help='epoch sae')
parser.add_argument('--epochs_CG',type=int,default=250,help='epoch CG')
parser.add_argument('--epochs_kan',type=int,default=150,help='epoch kan')
parser.add_argument('--epochs',type=int,default=70,help='out epoch')
parser.add_argument('--trend_length',type=int,default=64,help='trend input length')
parser.add_argument('--input_length',type=int,default=64,help='seasonal input length')
parser.add_argument('--use_data',type=str,default='Tetuan_City_power_consumption.csv',help='choose to use which data:Tetuan_City_power_consumption.csv,all')
parser.add_argument('--model_save',type=str,default=None,help='model save path')
parser.add_argument('--div_choose',type=str,default='fourier',choices=['l1','fourier','wavelet'],help='choose to use which div function')
parser.add_argument('--fourier_order',type=float,default=128,help='fourier order')
parser.add_argument('--lr',type=float,default=2e-5,help='learning rate')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--train_out_method',type=str,default='add',choices=['matrix','gate','add','FiLE'],help='train out method')
parser.add_argument('--plot_separate',type=bool,default=False,help='plot separate flag')
parser.add_argument('--train_trend_flag',type=bool,default=False,help='train trend flag')
parser.add_argument('--train_seasonal_flag',type=bool,default=False,help='train seasonal flag')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate')
args = parser.parse_args()
# print(args)

def data_set(args=args):
    train_size = args.train_size
    val_size = args.val_size
    step_size = args.out_length
    use_data_flag = args.use_data
    if use_data_flag is None:
        data_x, data_y = get_data(month=args.month,normalize_flag=args.normalize_flag)
    elif use_data_flag=='all':
        x3,y3=get_data(3)
        x4,y4=get_data(4)
        x5,y5=get_data(5)
        x6,y6=get_data(6)
        x7,y7=get_data(7)
        x8,y8=get_data(8)
        data_x,data_y=pd.concat((x3,x4,x5,x6,x7,x8),axis=0),pd.concat((y3,y4,y5,y6,y7,y8),axis=0)
    elif use_data_flag=='Tetuan_City_power_consumption.csv':
        data_x, data_y = read_csv(args.use_data)
    else:
        data_x,data_y=read_csv1(args.use_data)
    data_x,_,_=normalize(data_x)
    data_y,max,min=normalize3(data_y)
    data_x=data_x[:-step_size]
    data_y=data_y[step_size:]
    length = len(data_x)
    train_x_len = int(length * train_size)
    val_x_len = int(length * (train_size + val_size))
    train_x = data_x[:train_x_len]
    train_y = data_y[:train_x_len]
    val_x = data_x[train_x_len:val_x_len]
    val_y = data_y[train_x_len:val_x_len]
    test_x = data_x[val_x_len:]
    test_y = data_y[val_x_len:]
    print(f'train_x:{train_x.shape},train_y:{train_y.shape},test_x:{test_x.shape},test_y:{test_y.shape}')
    return torch.Tensor(train_x.to_numpy()), torch.Tensor(train_y.to_numpy()), torch.Tensor(val_x.to_numpy()), \
        torch.Tensor(val_y.to_numpy()), torch.Tensor(val_x.to_numpy()), torch.Tensor(val_y.to_numpy()),max.item(),min.item()

def t_stsakan():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x, train_y, val_x, val_y, test_x, test_y ,max,min = data_set()
    # train_x=train_x[:,-1:]
    # val_x=val_x[:,-1:]
    # test_x=test_x[:,-1:]
    time_start = time.time()
    model = stsakan(input_size_s=train_x.shape[1] , input_size_t=1, hidden_size=args.hidden_size, output_size=args.out_length,
                    input_length=args.input_length, out_length=args.out_length, num_layers=args.num_layers,
                    kernel_size=3, padding=1, lr=args.lr, args=args, stride=1, device=device)
    model.fit(train_x, train_y, val_x, val_y, epochs=args.epochs, batch_size=args.batch_size)
    time_end = time.time()
    if args.model_save is  None:
        path=f'stsakan{args.month}月_ol_{args.out_length}'
    else:
        path = f'{args.model_save}_ol_{args.out_length}'

    model.save_model(path)
    print(f'{path}训练完成，time cost:{(time_end-time_start)/60:.4f}minutes')
    predict, true = model.test(test_x.to(device), test_y.to(device),max,min)
    # predict1, true1 = model.predict(test_x.to(device), test_y.to(device))
    # 使用 os.path.join 构建路径
    result_dir = os.path.join('data', 'result')
    os.makedirs(result_dir, exist_ok=True)  # 确保目录存在

    # 添加下划线作为分隔符
    np.savetxt(os.path.join(result_dir, f'{path}_predict.csv'), predict.detach().cpu().numpy(), delimiter=',')
    np.savetxt(os.path.join(result_dir, f'{path}_true.csv'), true.detach().cpu().numpy(), delimiter=',')

def read_stsakan():
    path='best_model_fourier1.2_loss0.06170661002397537'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x, train_y, val_x, val_y, test_x, test_y = data_set()
    model=torch.load('model2/'+path+'.pth')
    predict, true = model.predict(test_x.to(device), test_y.to(device))
if __name__ == '__main__':
    # read_stsakan()
    t_stsakan()