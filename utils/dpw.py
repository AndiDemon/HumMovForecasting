# import pickle
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
# import cv2
import os
import glob

import torch
from scipy.spatial import distance


class Get_3dpw():
    def __init__(self, folder='../datasets/3dpw/sequenceFiles/'):
        self.folder = folder
        self.action_train = sorted(filter(os.path.isfile, glob.glob(self.folder + 'train' + '/*')))
        self.action_val = sorted(filter(os.path.isfile, glob.glob(self.folder + 'validation' + '/*')))
        self.action_test = sorted(filter(os.path.isfile, glob.glob(self.folder + 'test' + '/*')))

    def normalize_to_one(x):
        x = x / 1000
        return float(x)

    def forward(self, ignored_key=2):
        train = []
        val = []
        test = []

        # Training data
        print("==============Training Data===============")
        for _ in self.action_train:
            print("action = ", _)
            df = pd.read_pickle(_)
            for sex in range(len(df['poses2d'])):
                for frame in range(len(df['poses2d'][sex])):
                    keypoints = []
                    for keypoint in range(len(df['poses2d'][sex][0][0]) - ignored_key):
                        x = df['poses2d'][sex][frame][0][keypoint]
                        y = df['poses2d'][sex][frame][1][keypoint]
                        keypoints.append([x, y])
                    train.append(np.array(keypoints).flatten())

        # train = np.array(train).flatten()
        print("TRAINING DATA LENGTH = ", len(train))

        # Validation data
        print("==============Validation Data===============")
        for _ in self.action_val:
            print("action = ", _)
            df = pd.read_pickle(_)
            for sex in range(len(df['poses2d'])):
                for frame in range(len(df['poses2d'][sex])):
                    keypoints = []
                    for keypoint in range(len(df['poses2d'][sex][0][0]) - ignored_key):
                        x = df['poses2d'][sex][frame][0][keypoint]
                        y = df['poses2d'][sex][frame][1][keypoint]
                        keypoints.append([x, y])
                    val.append(np.array(keypoints).flatten())

        print("VALIDATION DATA LENGTH = ", len(val))

        # Testing data
        print("==============Testing Data===============")
        for _ in self.action_test:
            print("action = ", _)
            df = pd.read_pickle(_)
            for sex in range(len(df['poses2d'])):
                for frame in range(len(df['poses2d'][sex])):
                    keypoints = []
                    for keypoint in range(len(df['poses2d'][sex][0][0]) - ignored_key):
                        x = df['poses2d'][sex][frame][0][keypoint]
                        y = df['poses2d'][sex][frame][1][keypoint]
                        keypoints.append([x, y])
                    test.append(np.array(keypoints).flatten())

        print("TEST DATA LENGTH = ", len(test))

        return np.array(train), np.array(val), np.array(test)


class connect_3DPW():
    def __init__(self, folder='../datasets/3dpw/sequenceFiles/'):
        self.folder = folder
        self.action_train = sorted(filter(os.path.isfile, glob.glob(self.folder + 'train' + '/*')))
        self.action_val = sorted(filter(os.path.isfile, glob.glob(self.folder + 'validation' + '/*')))
        self.action_test = sorted(filter(os.path.isfile, glob.glob(self.folder + 'test' + '/*')))
        self.connection = [[1, 14, 15],
                           [2, 5, 8, 11],
                           [1, 3],
                           [2, 4],
                           [3],
                           [1, 6],
                           [5, 7],
                           [6],
                           [1, 9],
                           [8, 10],
                           [9],
                           [1, 12],
                           [1, 13],
                           [12],
                           [0],
                           [0]]

    def calc_movement(self, A, B):
        dist = distance.euclidean(A, B)
        ang = np.arccos(1 - distance.cosine(A, B))
        return [dist, ang]

    def forward(self, ignored_key=2):
        train = []
        val = []
        test = []

        # Training data
        print("==============Training Data===============")
        for _ in self.action_train:
            print("action = ", _)
            df = pd.read_pickle(_)
            for sex in range(len(df['poses2d'])):
                for frame in range(len(df['poses2d'][sex])):
                    connect = []
                    for connection in self.connection:
                        points = []
                        for point in connection:
                            x_temp = df['poses2d'][sex][frame][0][point]
                            y_temp = df['poses2d'][sex][frame][1][point]
                            points.append([x_temp, y_temp])
                        points = np.array(points)
                        connect.append(np.mean(points, axis=0))
                    train.append(np.array(connect).flatten())

        print("TRAINING DATA LENGTH = ", len(train))

        # Validation data
        print("==============Validation Data===============")
        for _ in self.action_val:
            print("action = ", _)
            df = pd.read_pickle(_)
            for sex in range(len(df['poses2d'])):
                for frame in range(len(df['poses2d'][sex])):
                    connect = []
                    for connection in self.connection:
                        points = []
                        for point in connection:
                            x_temp = df['poses2d'][sex][frame][0][point]
                            y_temp = df['poses2d'][sex][frame][1][point]
                            points.append([x_temp, y_temp])
                        points = np.array(points)
                        connect.append(np.mean(points, axis=0))
                    val.append(np.array(connect).flatten())

        print("VALIDATION DATA LENGTH = ", len(val))

        # Testing data
        print("==============Testing Data===============")
        for _ in self.action_test:
            print("action = ", _)
            df = pd.read_pickle(_)
            for sex in range(len(df['poses2d'])):
                for frame in range(len(df['poses2d'][sex])):
                    connect = []
                    for connection in self.connection:
                        points = []
                        for point in connection:
                            x_temp = df['poses2d'][sex][frame][0][point]
                            y_temp = df['poses2d'][sex][frame][1][point]
                            points.append([x_temp, y_temp])
                        points = np.array(points)
                        connect.append(np.mean(points, axis=0))
                    test.append(np.array(connect).flatten())

        print("TEST DATA LENGTH = ", len(test))

        return np.array(train), np.array(val), np.array(test)


def main():
    # data = Get_3dpw(get=True)
    # train, val, test = data.forward()
    # print(train.shape)
    # print(val.shape)
    # print(test.shape)
    # print(train[12])

    data = connect_3DPW(folder='../../datasets/3dpw/sequenceFiles/')
    train, val, test = data.forward()
    print(train.shape)
    print(val.shape)
    print(test.shape)
    print(train[12])

    # folder = '../datasets/3dpw/sequenceFiles/'
    # image_folder = '../datasets/3dpw/imageFiles/'
    # state = 'test/'
    # action = 'downtown_arguing_00'
    # gender = 0  # f = 0, m = 1
    #
    # # Training Files
    # df = pd.read_pickle(folder + state + action + '.pkl')
    # print(df.keys())
    # print(df['poses'][0].shape)
    # print(df['genders'][1])
    # print(df['img_frame_ids'][-1])
    # print(df['poses2d'][gender][0][0].shape)
    # # flat = (np.array(df['poses2d'][gender][0])).flatten()
    # print(df['poses2d'][gender][0].shape)
    # # poses2d[gender, frame_id, xy and probability, coordinate]
    #
    # imagename = sorted(filter(os.path.isfile, glob.glob(image_folder + action + '/*')))
    # for i in range(len(df['img_frame_ids'])):
    #     # fig = plt.figure()
    #     # x = np.array(df['poses2d'][0][i][0])
    #     # y = np.array(df['poses2d'][0][i][1])
    #     #
    #     # plt.scatter(x, y)
    #     # plt.gca().invert_yaxis()
    #     # plt.savefig(folder + 'poses2d/' + str(i) + '.png')
    #
    #     image = cv2.imread(imagename[i])
    #     color_f = (255, 0, 0)
    #     color_m = (0, 0, 255)
    #     thickness = 8
    #     # font
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #
    #     # fontScale
    #     fontScale = 1
    #
    #     # Using cv2.circle() method
    #     # Draw a circle with blue line borders of thickness of 2 px
    #     for j in range(len(df['poses2d'][gender][i][0])):
    #         x_f = int(df['poses2d'][gender][i][0][j])
    #         y_f = int(df['poses2d'][gender][i][1][j])
    #
    #         x_m = int(df['poses2d'][gender+1][i][0][j])
    #         y_m = int(df['poses2d'][gender+1][i][1][j])
    #         # print('x = ', x, ', y = ', y)
    #         # image = cv2.circle(image, (x_f, y_f), 4, color_f, thickness)
    #         image = cv2.putText(image, str(j), (x_f, y_f), font, fontScale, color_f, 2, cv2.LINE_AA)
    #
    #         # image = cv2.circle(image, (x_m, y_m), 4, color_m, thickness)
    #         image = cv2.putText(image, str(j), (x_m, y_m), font, fontScale, color_m, 2, cv2.LINE_AA)
    #
    #     # Displaying the image
    #     cv2.imwrite(folder + 'poses2d/' + str(i) + '.png', image)


if __name__ == '__main__':
    main()
