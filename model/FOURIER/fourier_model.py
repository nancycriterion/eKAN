from sklearn.model_selection import train_test_split

from .fourier import fourier, mse_mpe
from data.read_data import get_data,train_test_split1
import torch

# if torch.cuda.is_available():
#     print("GPU is available!")
# else:
#     print("GPU is not available.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    x, y = get_data(3)
    size =8
    output_length=1
    # x_test,y_test=get_data(4)
    x_train, x_test, y_train, y_test = train_test_split1(x, y, test_size=0.2)
    # print(x_train.shape)
    model = fourier(num_inputs=x.shape[1],output_length=output_length)
    model = model.to(device)  # 将模型迁移到 GPU
    model.setup_optimizer()
    model.fit(x_train, y_train, epochs=100, batch_size=size)
    predict = model.predict(x_test)
    # predict=predict.reshape(-1,1)
    # print(predict)
    # print(predict.shape,y_test.shape)
    mse, mpe, mae = mse_mpe(predict, y_test, batch_size=size,output_length=output_length)
    print('mse=', mse, f'\nmpe={mpe:.2f}%\nmae={mae:.2f}')
    return predict


if __name__ == '__main__':
    main()
