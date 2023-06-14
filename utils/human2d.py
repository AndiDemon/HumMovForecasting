#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_to_one(x):
    x = x / 1000
    return float(x)


def reverse_norm(arr):
    a = []
    for i in arr:
        a.append(i * 1000)
    return np.array(a)


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
        _source = torch.tensor(self.source[index], dtype=torch.float16)
        _target = torch.tensor(self.target[index], dtype=torch.float16)
        return _source, _target


class Human36M():
    def __init__(self, folder='../data/', cam_angle=(0, 1, 2, 3), motion='Walking'):
        self.folder = folder
        self.all_dirname = folder + 'allmotions/'
        self.training_subjects = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']  #
        # self.valid_subjects = ['S5']
        self.testing_subjects = ['S5']

        self.data = np.load(folder + 'data_2d_h36m_gt/positions_2d.npy', allow_pickle=True)
        self.cam_angle = cam_angle
        self.keypoints = 16
        self.motion = motion

    def sliding_window(self, data, input_n=25, output_n=25):
        source_loop = input_n  # 50
        target_loop = output_n  # 25
        src = []  # np.zeros((len(data), 25, 32))
        trg = []  # np.zeros((len(data), 25, 32))
        source = []
        target = []
        start = True
        length_src = source_loop
        length_trg = target_loop
        length_data = len(data)
        for i in range(len(data) - (source_loop + target_loop)):
            src = []
            trg = []
            for s in range(source_loop):
                src.append(data[i + s])
            for t in range(target_loop):
                trg.append(data[i + source_loop + t])
            source.append(src)
            target.append(trg)

        return np.array(source), np.array(target)

    def get_data(self, state='train'):
        combined = []
        # (dict,     dict,       list,  list,      list, list * float32)
        # (subject, motion, cam_angle, frame, keypoints, coordinate * 2)
        if state == 'train':
            for sub in self.training_subjects:
                motion_list = sorted(os.listdir(self.all_dirname + sub))
                matching = [s for s in motion_list if self.motion in s]
                for mot in matching:
                    for cam in self.cam_angle:
                        for frame in range(len(self.data.item()[sub][mot][cam])):
                            s = []
                            for key in range(self.keypoints):
                                source = normalize_to_one(
                                    self.data.item()[sub][mot][cam][frame][key][0]), normalize_to_one(
                                    self.data.item()[sub][mot][cam][frame][key][1])
                                s.append(source)
                            s = np.array(s).flatten()
                            combined.append(s)

        elif state == 'test':
            for sub in self.testing_subjects:
                motion_list = sorted(os.listdir(self.all_dirname + sub))
                matching = [s for s in motion_list if self.motion in s]
                for mot in matching:
                    for cam in self.cam_angle:
                        for frame in range(len(self.data.item()[sub][mot][cam])):
                            s = []
                            for key in range(self.keypoints):
                                source = normalize_to_one(
                                    self.data.item()[sub][mot][cam][frame][key][0]), normalize_to_one(
                                    self.data.item()[sub][mot][cam][frame][key][1])
                                s.append(source)
                            s = np.array(s).flatten()
                            combined.append(s)
        return self.sliding_window(np.array(combined), input_n=25, output_n=25)


def main():
    human36m = Human36M(folder='../data/', cam_angle=[1], motion='WalkTogether')
    train = human36m.get_data('train')
    test = human36m.get_data('test')

    reversed = reverse_norm(test)
    print(reversed.shape)
    print(reversed[0])


if __name__ == '__main__':
    main()
