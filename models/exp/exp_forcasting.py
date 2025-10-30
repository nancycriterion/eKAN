import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from models.utils.losses import *
from .exp_base import Exp_Basic
from data.read_data import *
from tqdm import tqdm
class exp_forcasting(Exp_Basic):
    def __init__(self, args):
        super(exp_forcasting, self).__init__(args)

    def _build_model(self):
        model=self.model_dict[self.args.model_name].Model(self.args).float()
        return model
    # def _get_data(self):
    #     train_x, train_y, val_x, val_y, test_x, test_y = data_set()
    #     return train_x, train_y, val_x, val_y, test_x, test_y
    def _get_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
    def data_loader(self,x):
        x_loader=[]
        for i in range(len(x)-self.args.batch_size+1-self.args.batch_length):
            x_temp=[]
            for j in range(self.args.batch_size):
                x_temp.append(x[i+j:i+j+self.args.batch_length])
            x_temp=torch.stack(x_temp,dim=0)
            x_loader.append(x_temp)
        x_loader=torch.stack(x_loader,dim=0)
        return x_loader
    def train(self,train_x,train_y,val_x,val_y):
        model_optim=self._get_optimizer()
        criterion=self._select_criterion(loss_name=self.args.loss_name)
        mse=nn.MSELoss()
        train_y=self.data_loader(train_x[self.args.out_length:])
        train_x=self.data_loader(train_x[:-self.args.out_length])
        val_x=self.data_loader1(val_x)
        val_y=self.data_loader1(val_y)
        for epoch in range(self.args.num_epochs):
            self.model.train()
            train_loss=0
            val_loss=0
            for i, (x, y) in tqdm(enumerate(zip(train_x, train_y)),total=len(train_x)):
                x=x.float().to(self.device)#(batch_size,seq_len,input_dim)
                y=y.float().to(self.device)#(batch_size,seq_len,output_dim)
                model_optim.zero_grad()
                dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                output=self.model(x,None,dec_inp,None)
                output=output[:,-self.args.pred_len:,-1:]
                y=y[:,-self.args.pred_len:,-1:]
                loss=criterion(output,y)
                train_loss+=loss
                loss.backward()
                model_optim.step()
            self.model.eval()
            for i, (x, y) in tqdm(enumerate(zip(val_x, val_y)),total=len(val_x)):
                x=x.float().to(self.device)
                y=y.float().to(self.device)
                B, _, C = x.shape
                dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
                dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
                output=self.model(x,None,dec_inp,None)
                output=output[:,-self.args.pred_len:,-1:]
                y=y[:,-self.args.pred_len:,-1:]
                val_loss+=criterion(output,y)
            if (epoch+1)%1==0:
                print('Epoch:{}, Train Loss:{:.4f}, Val Loss:{:.4f}'.format(epoch+1,train_loss,val_loss))
        return self.model

    def data_loader1(self,x):
        x_loader=[]
        for i in range(len(x)-1+1-self.args.batch_length):
            x_temp=[]
            for j in range(1):
                x_temp.append(x[i+j:i+j+self.args.batch_length])
            x_temp=torch.stack(x_temp,dim=0)
            x_loader.append(x_temp)
        x_loader=torch.stack(x_loader,dim=0)
        return x_loader
    def test(self,test_x,test_y,max,min):
        criterion=self._select_criterion(loss_name=self.args.loss_name)
        mse=nn.MSELoss()
        test_x=self.data_loader1(test_x)
        test_y=self.data_loader1(test_y)
        self.model.eval()
        test_loss=0
        out=[]
        true=[]
        for i, (x, y) in enumerate(zip(test_x, test_y)):
            x=x.float().to(self.device)
            y=y.float().to(self.device)
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            output = self.model(x, None, dec_inp, None)
            output=output[:,-self.args.pred_len:,-1:].reshape(output.shape[0],-1)
            y=y[:,-self.args.pred_len:,:]
            y=y[:,:,-1:].reshape(y.shape[0],-1)
            test_loss+=criterion(output,y)
            out.append(output)
            true.append(y)
        out=torch.cat(out,dim=0)
        true=torch.cat(true,dim=0)
        true=denormalize3(true,max,min)
        out=denormalize3(out,max,min)
        print('{}Test Loss:{}'.format(self.args.model_name,test_loss),f'\npredict shape:{out.shape}true shape:{true.shape}')
        self.predict_loss(out.to('cpu').detach().numpy(), true.to('cpu').detach().numpy())
        return out,true
    def predict_loss(self,predict,y_test_true):
        mse=np.mean((predict-y_test_true)**2)
        mae=np.mean(np.abs(predict-y_test_true))
        rmse=np.sqrt(mse)
        mask = y_test_true != 0
        mape = np.mean(np.abs((predict[mask] - y_test_true[mask]) / y_test_true[mask])) * 100
        smape=np.mean(np.abs(predict-y_test_true)/(np.abs(predict)+np.abs(y_test_true)))*2*100
        nrmse=rmse/np.std(y_test_true)*100
        print(f'mse:{mse},mae:{mae},rmse:{rmse},mape:{mape:.4f}%,smape:{smape:.4f}%,nrmse:{nrmse:.4f}%')
        self.predict_3D(predict, y_test_true)
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

