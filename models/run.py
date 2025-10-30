import argparse
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from models.exp import *
from models.exp.exp_forcasting import exp_forcasting
from models.timenet import *
from set import data_set
from data.read_data import *
parser = argparse.ArgumentParser()
# parser.add_argument('--model',type=str,default='timenet',help='model name')
#加入task_name,seq_len,label_num,pred_num,e_layers,d_model,c_out,device,learning_rate,loss_name,num_epochs,pred_len
#is_training,batch_size.batch_length,use_gpu,model_name,top_k,d_ff,num_kernels,enc_in,embed,freq,dropout
parser.add_argument('--task_name',type=str,default='task_name',help='task name')
parser.add_argument('--label_num',type=int,default=1,help='label number')
parser.add_argument('--pred_num',type=int,default=1,help='prediction number')
parser.add_argument('--e_layers',type=int,default=2,help='encoder layers')
parser.add_argument('--d_model',type=int,default=6,help='embedding dimension')
parser.add_argument('--c_out',type=int,default=1,help='output dimension')
parser.add_argument('--device',type=str,default='cuda:0',help='device')
parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
parser.add_argument('--loss_name',type=str,default='MSE',choices=['MSE','MAPE','MASE','SMAPE'],help='loss function name')
parser.add_argument('--num_epochs',type=int,default=10,help='number of epochs')
parser.add_argument('--is_training',type=bool,default=True,help='is training')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--batch_length',type=int,default=64,help='batch length')
parser.add_argument('--use_gpu',type=bool,default=True,help='use gpu')
parser.add_argument('--model_name',type=str,default='timemixer',
                    choices=['timenet','non_stationary_transformer','autoformer','fedformer','itransformer','timemixer'],help='model name')
parser.add_argument('--top_k',type=int,default=3,help='top k')
parser.add_argument('--d_ff',type=int,default=2048,help='feedforward dimension')
parser.add_argument('--num_kernels',type=int,default=3,help='number of kernels')
parser.add_argument('--enc_in',type=int,default=6,help='encoder input dimension')
parser.add_argument('--embed',type=str,default='timeF',choices=['timeF','timeF_delta','timeF_delta2'],help='embedding type')
parser.add_argument('--freq',type=str,default='h',choices=['d','h','t'],help='frequency')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout rate')
parser.add_argument('--use_data',type=str,default='Tetuan_City_power_consumption.csv',help='use data')
parser.add_argument('--month',type=int,default=3,help='month of data')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--normalize_flag',type=bool,default=True,help='normalize flag')
parser.add_argument('--train_size',type=float,default=0.7,help='train size')
parser.add_argument('--val_size',type=float,default=0.15,help='val size')
parser.add_argument('--out_length',type=int,default=48,help='output length')
parser.add_argument('--pred_len',type=int,default=parser.get_default('out_length'),help='prediction length')
parser.add_argument('--label_len', type=int, default=24, help='start token length')
parser.add_argument('--factor', type=int, default=0.3, help='attn factor')
parser.add_argument('--n_heads', type=int, default=6, help='num of heads')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--use_data_flag',type=str,default=None,help='use data flag')
parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--channel_independence', type=int, default=0,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
args = parser.parse_args()

if __name__ =='__main__':
    Exp=exp_forcasting
    

    x_train,y_train,x_val,y_val,x_test,y_test,max,min=data_set(args=args)
    x_train=torch.cat([x_train,x_train[:,-1:]],dim=1)
    x_val=torch.cat([x_val,x_val[:,-1:]],dim=1)
    x_test=torch.cat([x_test,x_test[:,-1:]],dim=1)


    if args.is_training:
        exp = Exp(args)
        print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
        time_start = time.time()
        exp.train(x_train[:,:],y_train,x_val[:,:],y_val)
        time_end = time.time()
        print('training time cost:',(time_end-time_start)/60,'min')
        print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        predict,true=exp.test(x_test[:,:],y_test,max,min)

        result_dir = os.path.join('data', 'result2')
        os.makedirs(result_dir, exist_ok=True)  # 确保目录存在

        # 添加下划线作为分隔符
        np.savetxt(os.path.join(result_dir, f'{args.model_name}_ol_{args.out_length}_predict.csv'), predict.detach().cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(result_dir, f'{args.model_name}_ol_{args.out_length}_true.csv'), true.detach().cpu().numpy(), delimiter=',')
    else:
        pass