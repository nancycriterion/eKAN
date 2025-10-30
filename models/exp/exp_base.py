import os
import torch
from models.timenet import timenet
from models.autoformer import Autoformer
from models.non_stationary_Transformer import Non_stationary_Transformer
from models.FEDformer import FEDformer
from models.itransformer import itransformer
from models.timemixer import timemixer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'timenet': timenet,
            'autoformer': Autoformer,
            'non_stationary_transformer': Non_stationary_Transformer,
            'fedformer':FEDformer,
            'itransformer':itransformer,
            'timemixer':timemixer
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print('Use GPU')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self,x_train, y_train, x_val, y_val):
        pass

    def test(self,x_test, y_test):
        pass

    def predict_loss(self, predict, y_test_true):
        pass

    def predict_3D(self, predict, y_test_true):
        pass