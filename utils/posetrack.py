#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import glob
import numpy as np
import cv2
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import sys
import torch
from torch.utils.data import Dataset, DataLoader


class Posetrack():
    def __init__(self, folder='../../datasets/posetrack/'):
        self.folder = folder

        self.train_in = 'posetrack_train_in.json'
        self.train_out = 'posetrack_train_out.json'
        self.train_mask_in = 'posetrack_train_masks_in.json'
        self.train_mask_out = 'posetrack_train_masks_out.json'

        self.val_in = 'posetrack_valid_in.json'
        self.val_out = 'posetrack_valid_out.json'
        self.val_mask_in = 'posetrack_valid_masks_in.json'
        self.val_mask_out = 'posetrack_valid_masks_out.json'

        self.test_in = 'posetrack_test_in.json'
        self.test_out = 'posetrack_test_out.json'
        self.test_mask_in = 'posetrack_test_masks_in.json'
        self.test_mask_out = 'posetrack_test_masks_out.json'

    def get_data(self, state='train'):
        in_ = []
        out_ = []
        mask_in = []
        mask_out = []
        data_in = []
        data_out = []
        data_mask_in = []
        data_mask_out = []

        if state == 'train':
            data_in = pd.read_json(self.folder + self.train_in)
            data_out = pd.read_json(self.folder + self.train_out)
            data_mask_in = pd.read_json(self.folder + self.train_mask_in)
            data_mask_out = pd.read_json(self.folder + self.train_mask_out)
        elif state == 'val':
            data_in = pd.read_json(self.folder + self.val_in)
            data_out = pd.read_json(self.folder + self.val_out)
            data_mask_in = pd.read_json(self.folder + self.val_mask_in)
            data_mask_out = pd.read_json(self.folder + self.val_mask_out)
        elif state == 'test':
            data_in = pd.read_json(self.folder + self.test_in)
            # data_out = pd.read_json(self.folder + self.test_out)
            data_mask_in = pd.read_json(self.folder + self.test_mask_in)
            # data_mask_out = pd.read_json(self.folder + self.test_mask_out)

        # print(data_mask_in[0][0][0])
        # print(data_in.shape)

        # source, target
        for i in data_in:
            for j in data_in[i]:
                if j != None:
                    time = []
                    for k in range(len(j)):
                        time.append(j[k])
                    in_.append(time)

        for i in data_out:
            for j in data_out[i]:
                if j != None:
                    time = []
                    for k in range(len(j)):
                        time.append(j[k])
                    out_.append(time)

        for i in data_mask_in:
            for j in data_mask_in[i]:
                if j != None:
                    time = []
                    for k in range(len(j)):
                        time.append(j[k])
                    mask_in.append(time)

        for i in data_mask_out:
            for j in data_mask_out[i]:
                if j != None:
                    time = []
                    for k in range(len(j)):
                        time.append(j[k])
                    mask_out.append(time)

        # # mask
        # for i in range(data_mask_in.shape[1]):
        #     for j in range(data_mask_in.shape[0]):
        #         mask_in.append(np.array(data_mask_in[i][j]))
        # for i in range(data_mask_out.shape[1]):
        #     for j in range(data_mask_out.shape[0]):
        #         mask_out.append(np.array(data_mask_out[i][j]))

        in_ = np.array(in_)
        out_ = np.array(out_)
        mask_in = np.array(mask_in)
        mask_out = np.array(mask_out)
        return in_, out_, mask_in, mask_out


class TimeSeriesDataSet(Dataset):
    """
    This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
    getting bogged down by the preprocessing
    """

    def __init__(self, source, target):
        self.source = source
        self.target = target
        if len(self.source) != len(self.target):
            raise Exception("The length of source does not match the length of target")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        # print(self.source[index].shape)

        _source = torch.tensor(self.source[index])
        _target = torch.tensor(self.target[index])
        return _source, _target


def main():
    # folder = '../../datasets/posetrack/'
    # data_in = pd.read_json(folder + 'posetrack_train_in.json')
    # data_out = pd.read_json(folder + 'posetrack_train_out.json')

    # (video, person, time_step, coordinate of joints)
    #
    # print(data_in[0].shape)
    # print(data_out[0].shape)

    posetrack = Posetrack()
    train_in, train_out, train_mask_in, train_mask_out = posetrack.get_data('train')
    # val_in, val_out, val_mask_in, val_mask_out = posetrack.get_data('val')
    # test_in, test_out, test_mask_in, test_mask_out = posetrack.get_data('test')
    # print(len(train_in[0]))
    print(train_in.shape)
    print(train_out.shape)
    print(train_mask_in.shape)
    # print(train_in[21][306])
    # print(test_in.shape)


if __name__ == '__main__':
    main()
