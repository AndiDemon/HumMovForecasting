import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from utils import Torch_Transformer as t
from utils.human2d import Human36M, TimeSeriesDataSet

import torch
from torch import nn, Tensor
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# heatmap plot
import seaborn as sns

# STS-GCN model
from Model.STSGCN import *
# RNN Model
from Model.LSTM import *
from Model.Attention import TS_TSSA
# 3DPW dataset
from utils.dpw import *

# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# device = 'cpu'
if torch.cuda.is_available():
    print("cuDNN available")
    cudnn.benchmark = True


def train_loop(model, opt, loss_fn, dataloader):
    """
    Args:
        model: untrained model
        opt: optimizer
        loss_fn: loss function
        dataloader: train input and expected output

    Returns: train loss

    """
    model.train()
    total_loss = 0

    for (batch, (src, trg)) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)


        trg_expected = trg[:, :, :]  # batch_size, frame, keypoints x and y

        # Get mask to mask out the next words
        sequence_length = trg.size(1)
        pred = model(src)
        pred = pred[:, :, :]

        loss = torch.sqrt(loss_fn(pred.to(torch.float32), trg_expected.to(torch.float32))) \
               + loss_fn(pred[:, :, 6:9].to(torch.float32), trg_expected[:, :, 6:9].to(torch.float32)) * 4 \
               + loss_fn(pred[:, :, 12:15].to(torch.float32), trg_expected[:, :, 12:15].to(torch.float32)) * 4 \
               + loss_fn(pred[:, :, 20:21].to(torch.float32), trg_expected[:, :, 20:21].to(torch.float32)) * 4 \
               + loss_fn(pred[:, :, 26:27].to(torch.float32), trg_expected[:, :, 26:27].to(torch.float32)) * 4

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def val_loop(model, loss_fn, dataloader):
    """

    Args:
        model: trained model
        loss_fn: loss function
        dataloader: input and expected output

    Returns: validation loss

    """
    model.eval()
    total_loss = 0

    for (batch, (src, trg)) in enumerate(dataloader):
        # print("BATCH = ", batch)
        src, trg = src.to(device), trg.to(device)
        trg_expected = trg[:, :, :]  # batch_size, frame, keypoints x and y

        pred = model(src)
        pred = pred[:, :, :]

        loss = torch.sqrt(loss_fn(pred.to(torch.float32), trg_expected.to(torch.float32))) \
               + loss_fn(pred[:, :, 6:9].to(torch.float32), trg_expected[:, :, 6:9].to(torch.float32)) * 4 \
               + loss_fn(pred[:, :, 12:15].to(torch.float32), trg_expected[:, :, 12:15].to(torch.float32)) * 4 \
               + loss_fn(pred[:, :, 20:21].to(torch.float32), trg_expected[:, :, 20:21].to(torch.float32)) * 4 \
               + loss_fn(pred[:, :, 26:27].to(torch.float32), trg_expected[:, :, 26:27].to(torch.float32)) * 4

        # loss = t.mpjpe(pred.to(torch.float32), trg_expected.to(torch.float32))

        total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, model_dirname, motion=''):
    """

    Args:
        model: untrained model
        opt: optimizer
        loss_fn: loss function
        train_dataloader: train data
        val_dataloader: validation data
        epochs: number of epochs or iteration
        model_dirname: destination to save the model
        motion: name of motion

    Returns: Train loss, validation loss

    """
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        val_loss = val_loop(model, loss_fn, val_dataloader)
        val_loss_list += [val_loss]

        if (epoch % 5) == 0:
            # save model
            torch.save(model, model_dirname)

        print("EPOCH = ", epoch + 1, " motions = ", motion)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        # # if dataset=='3dpw':
        # plt.xlabel('Epochs')
        # # plt.xlim((0, len(testloss)))
        # plt.ylabel('RMSE')
        # # plt.title("Training Loss")
        # val_line = plt.plot(val_loss_list, label='Testing', color='blue')
        # train_line = plt.plot(train_loss_list, label='Training', color='red')
        # plt.legend([ 'Testing', 'Training'])
        # plt.savefig('./Testing/Human36M/special/3dpw_backward.eps') #'./Testing/Human36M/' + folder + '/train.png'

    return train_loss_list, val_loss_list


def predict(model, source, targets):
    """
    Args:
        model: The trained model
        source: input data
        targets: expected output from ground truth

    Returns: Prediction Results, MPJPE, input data, MPJPE on keypoints, expected output.

    """
    model.eval()
    total_loss = 0
    testloss, pred, src, gt = [], [], [], []
    keypoints = 16
    keypoint_loss = np.empty((keypoints, source.shape[0]))
    frame = 24

    for i in range(25):
        src.append(source[0][i][:])
    for i in range(source.shape[0]):

        x = torch.tensor(source[i:i + 1, :, :], dtype=torch.float16).to(device)
        y = torch.tensor(targets[i:i + 1, :, :], dtype=torch.float16).to(device)
        result = model(x)

        loss = rmse_ex(result[0, 0:frame, :].cpu().detach().numpy(), targets[i, 0:frame, :])
        testloss.append(loss)
        pred.append(result[0, frame, :].cpu().detach().numpy())
        gt.append(targets[i, frame, :])
        src.append(source[i, frame, :])

        for j in range(keypoints):
            k_loss = t.mpjpe(result[0, frame, j * 2:(j * 2) + 1],
                             torch.tensor(targets[i, frame, j * 2:(j * 2) + 1]).to(device))
            keypoint_loss[j][i] = k_loss.cpu().detach().numpy()

    pred = np.array(pred)
    key_loss = np.mean(keypoint_loss, axis=1)

    return pred, testloss, src, key_loss, gt


def main():
    print(torch.cuda.is_available())
    EPOCHS = 200
    BATCH_SIZE = 64  # 8
    MODEL_DIR = "./checkpoints/hum36m/"
    MOTIONS = ['Walking', 'WalkDog', 'WalkTogether', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
               'Posing', 'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'Directions']

    # model assigned
    model_name = "Attention"
    num_layers = 1
    d_model = 32
    embed_dim = 32
    dff = 256
    num_heads = 4
    dropout_rate = 0.1
    model = TS_TSSA(d_model, embed_dim, num_heads, num_layers, output_n=25, input_n=25,
                           dropout=dropout_rate, dff=dff, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)  #
    loss_fn = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters in the model:", total_params)

    # model_name = "GRU"
    # model = GRUModel(input_dim=embed_dim, hidden_dim=dff, layer_dim=num_layers, output_dim=embed_dim, dropout_prob=dropout_rate, output_window=25, device='cuda:0')

    for mot in MOTIONS:
        identity = model_name + mot
        model_dirname = MODEL_DIR + identity + '.pt'

        print("--------------------LOADING DATA----------------------")
        human36m = Human36M(folder='./data/', cam_angle=[1], motion=mot)
        train_data_source, train_data_target = human36m.get_data('train')
        val_data_source, val_data_target = human36m.get_data('test')
        test_data_source, test_data_target = human36m.get_data('test')

        print("--------------------DATA LOADED----------------------")
        print("--------------------", mot, "----------------------")
        print(train_data_source.shape)
        print(train_data_target.shape)
        print("==========test==========")
        print(test_data_source.shape)
        print(test_data_target.shape)

        # The Dataloader class handles all the shuffles for you
        train_data = DataLoader(TimeSeriesDataSet(train_data_source, train_data_target), batch_size=BATCH_SIZE,
                                shuffle=True)
        val_data = DataLoader(TimeSeriesDataSet(val_data_source, val_data_target), batch_size=BATCH_SIZE,
                              shuffle=True)

        train_loss_list, val_loss_list = fit(model, opt, loss_fn, train_data, val_data, EPOCHS, model_dirname,
                                             motion=mot)

        # test predict
        pred, mpjpe, source, keypoint_loss, gt = predict(model, test_data_source, test_data_target)
        mpjve = t.mean_velocity_error(np.array(pred), np.array(gt))
        print("MPJPE average 1000msec (scale 0-1) = ", np.mean(mpjpe))
        print("MPJVE average 1000msec (scale 0-1) = ", np.mean(mpjve))
        print(keypoint_loss)


if __name__ == '__main__':
    main()
