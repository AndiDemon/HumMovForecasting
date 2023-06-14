import os
import time
import csv
import numpy as np
import pandas as pd
import math
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
from thop import profile

mpl.rc('figure', max_open_warning=0)
from utils import Torch_Transformer as t
import torch
from torch import nn, Tensor
import torch.utils.data
import torch.backends.cudnn as cudnn

# heatmap plot
import seaborn as sns

from utils.human2d import Human36M

# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# device = 'cpu'
if torch.cuda.is_available():
    print("cuDNN available")
    cudnn.benchmark = True


def normalize_to_one(x):
    x = x / 1000
    return float(x)


def return_normalization(x):
    x = x * 1000
    return x


def sliding_window(data):
    source_loop = 25
    target_loop = 25
    src = []  # np.zeros((len(data), 25, 32))
    trg = []  # np.zeros((len(data), 10, 32))
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


def predict(model, source, targets, _source=(), _targets=(), devices=device):
    global flops
    print('device = ', devices)
    model.eval()
    total_loss = 0
    testloss = []
    pred = [[], [], [], [], [], [], [], []]
    src = []
    gt = [[], [], [], [], [], [], [], []]
    form = [1, 3, 7, 9, 13, 17, 21, 24]
    keypoints = 16
    keypoint_loss = np.empty((keypoints, source.shape[0]))
    loss = 0
    all_time = []
    length_target = len(targets)
    torch.backends.cudnn.benchmark = True
    for i in range(25):
        pred.append(source[0][i][:])
        src.append(source[0][i][:])
    for i in range(length_target):
        start = time.time()
        x = torch.tensor(source[i:i + 1, :, :]).to(devices)
        y = torch.tensor(targets[i:i + 1, :, :]).to(devices)
        result = model(x)
        flops, params = profile(model, inputs=(x,))

        all_time.append(time.time() - start)
        # loss = t.p_mpjpe(result[0, 0:24, :].cpu().detach().numpy(), targets[i, 0:24, :])
        loss = t.rmse_ex(result[0, 23:24, :].cpu().detach().numpy(), targets[i, 23:24, :])
        testloss.append(loss)
        for j in range(len(form)):
            pred[j].append(result[0, form[j], :].cpu().detach().numpy())
            gt[j].append(targets[i, form[j], :])
        src.append(source[i, 24, :])

        # for j in range(keypoints):
        #     k_loss = return_normalization(t.mpjpe(torch.tensor(result[0, 24, j * 2:(j * 2) + 1].cpu().detach().numpy()),
        #                                           torch.tensor(targets[i, 24, j * 2:(j * 2) + 1])))
        #     keypoint_loss[j][i] = k_loss
    print("FLOPS:", flops)
    pred = np.array(pred)
    gt = np.array(gt)
    key_loss = np.mean(keypoint_loss, axis=1)
    print("average time taken = ", np.mean(np.array(all_time)))
    return pred, testloss, src, key_loss, gt


def accuracy_plot(x, x_label, y_label, model_dir):
    data = np.asarray(x).reshape(14, 16)

    # setting the parameter values
    annot = True

    # plotting the heatmap
    hm = sns.heatmap(data=data, xticklabels=x_label, yticklabels=y_label)

    # displaying the plotted heatmap
    plt.show()
    plt.savefig('./eval/plot/' + model_dir + '.eps', bbox_inches='tight')


def calc_testloss(combine, gt, mse_loss_fn=nn.MSELoss()):
    # separate the keypoints coordinate
    test_loss = []
    even = np.arange(0, 31, 2)
    combine = np.asarray(combine)
    gt = np.asarray(gt)
    organized = np.zeros((len(combine), 16, 2))
    organized_gt = np.zeros((len(gt), 16, 2))
    for row in range(len(organized)):
        i = 0
        for column in even:
            organized[row][i][0] = return_normalization(combine[row][column])
            organized[row][i][1] = return_normalization(combine[row][column + 1])
            i += 1

    for row in range(len(organized_gt)):
        i = 0
        for column in even:
            # print('i = ', i)
            organized_gt[row][i][0] = return_normalization(gt[row][column])
            organized_gt[row][i][1] = return_normalization(gt[row][column + 1])
            i += 1

    line = [[0, 1], [1, 2], [1, 5], [1, 8], [1, 12], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10],
            [10, 11], [12, 13], [13, 14], [14, 15]]

    # print(organized[0])
    organized = np.nan_to_num(organized)
    organized_gt = np.nan_to_num(organized_gt)
    # print(organized.shape)
    # print("prediction : ", organized[:, :, :])
    # print("target : ", organized_gt[:, :, :])
    mpjpe_loss = t.mpjpe(torch.tensor(organized[:, :, :]), torch.tensor(organized_gt[:, :, :]))
    mae_loss = t.mean_velocity_error(organized[:, :, :], organized_gt[:, :, :])
    mse_loss = mse_loss_fn(torch.tensor(organized[:, :, :]), torch.tensor(organized_gt[:, :, :]))

    for row in range(len(organized)):
        loss = t.mpjpe(torch.tensor(organized[row, :, :]), torch.tensor(organized_gt[row, :, :]))
        test_loss.append(loss.numpy())
    return mpjpe_loss, mae_loss, mse_loss, test_loss


def plot_humanposeGraph(ground_truth, prediction, motion, motion_count):
    even = np.arange(0, 31, 2)
    graph = list(np.arange(0, 80, 10))

    ground_truth = np.asarray(ground_truth)
    organized = np.zeros((len(ground_truth), 16, 2))

    prediction = np.asarray(prediction)
    organized_pred = np.zeros((len(prediction), 16, 2))

    for row in range(len(organized)):
        i = 0
        for column in even:
            organized[row][i][0] = return_normalization(ground_truth[row][column])
            organized[row][i][1] = return_normalization(ground_truth[row][column + 1])
            i += 1
    for row in range(len(organized_pred)):
        i = 0
        for column in even:
            organized_pred[row][i][0] = return_normalization(prediction[row][column])
            organized_pred[row][i][1] = return_normalization(prediction[row][column + 1])
            i += 1

    line = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [7, 0], [7, 8], [8, 9], [8, 13], [13, 14],
            [14, 15], [8, 10], [10, 11], [11, 12]]

    print("mpjpe = ", t.rmse_ex(organized_pred, organized))

    count = 0
    gt_all_pose = []
    pred_all_pose = []
    keypoint = 16
    gap_times_x = 15
    gap_times_y = 240
    fig = plt.figure()
    fig.set_size_inches(26, 6)
    for gt, pred in zip(organized, organized_pred):
        gt = gt.tolist()
        pred = pred.tolist()
        gt_pose = []
        pred_pose = []

        gap = gap_times_x * count
        gap_motion = gap_times_y * motion_count
        if count in graph:
            for pair in line:
                partA = pair[0]
                partB = pair[1]
                if gt[partA] and gt[partB]:
                    plt.plot([gt[partA][0] + gap, gt[partB][0] + gap],
                             [gt[partA][1] + gap_motion, gt[partB][1] + gap_motion], 'k:')
                    plt.plot([pred[partA][0] + gap, pred[partB][0] + gap],
                             [pred[partA][1] + gap_motion, pred[partB][1] + gap_motion], 'c-')
                    gt_pose.append([gt[partA], gt[partB]])
                    pred_pose.append([pred[partA], pred[partB]])

        gt_all_pose.append(gt_pose)
        pred_all_pose.append(pred_pose)
        count += 1
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('./eval/graph/' + motion + '.eps')


def data_openpose(motion, data_folder):
    data = pd.read_csv(data_folder, header=None)
    data = np.array(data)
    src = np.empty((len(data), len(data[0])))
    for i in range(len(data)):
        for key in range(32):
            src[i][key] = normalize_to_one(data[i][key])
    return src


def to_video_slide(combine, source, gt, get_video, save_video):
    # separate the keypoints coordinate
    even = np.arange(0, 31, 2)
    # print(even)
    combine = np.asarray(combine)
    source = np.asarray(source)
    gt = np.asarray(gt)
    organized = np.zeros((len(combine), 16, 2))
    organized_source = np.zeros((len(source), 16, 2))
    organized_gt = np.zeros((len(gt), 16, 2))
    for row in range(len(organized)):
        i = 0
        for column in even:
            organized[row][i][0] = return_normalization(combine[row][column])
            organized[row][i][1] = return_normalization(combine[row][column + 1])
            organized_source[row][i][0] = return_normalization(source[row][column])
            organized_source[row][i][1] = return_normalization(source[row][column + 1])
            i += 1

    for row in range(len(organized_gt)):
        i = 0
        for column in even:
            # print('i = ', i)
            organized_gt[row][i][0] = return_normalization(gt[row][column])
            organized_gt[row][i][1] = return_normalization(gt[row][column + 1])

            i += 1

    line = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [7, 0], [7, 8], [8, 9], [8, 13], [13, 14],
            [14, 15], [8, 10], [10, 11], [11, 12]]

    organized = np.nan_to_num(organized)
    organized_source = np.nan_to_num(organized_source)
    organized_gt = np.nan_to_num(organized_gt)

    cap = cv2.VideoCapture(get_video)
    while not cap.isOpened():
        cap = cv2.VideoCapture(get_video)
        cv2.waitKey(1000)
        print("Wait for the header")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter('./eval/video/' + save_video + '.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             25, size)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    cam_angle = 1
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # csv from the joints groundtruth
        if (count >= 0) & (count < len(organized)):
            for num in range(16):
                for pair in line:
                    partA = pair[0]
                    partB = pair[1]
                    # prediction
                    center_coordinates_one = (int(organized[count][partA][0]), int(organized[count][partA][1]))
                    center_coordinates_two = (int(organized[count][partB][0]), int(organized[count][partB][1]))

                    # source
                    source_coordinates_one = (
                        int(organized_source[count][partA][0]), int(organized_source[count][partA][1]))
                    source_coordinates_two = (
                        int(organized_source[count][partB][0]), int(organized_source[count][partB][1]))

                    # gt
                    gt_coordinates_one = (int(organized_gt[count][partA][0]), int(organized_gt[count][partA][1]))
                    gt_coordinates_two = (int(organized_gt[count][partB][0]), int(organized_gt[count][partB][1]))

                    if center_coordinates_one and center_coordinates_two:
                        cv2.line(frame, center_coordinates_one, center_coordinates_two, (0, 255, 0), 2,
                                 lineType=cv2.LINE_AA)  # green prediction
                        cv2.line(frame, source_coordinates_one, source_coordinates_two, (0, 0, 255), 2,
                                 lineType=cv2.LINE_AA)  # red source
                        cv2.line(frame, gt_coordinates_one, gt_coordinates_two, (252, 119, 3),
                                 2,
                                 lineType=cv2.LINE_AA)  # blue gt

            for num in range(16):
                center_coordinates = (int(organized[count][num][0]), int(organized[count][num][1]))
                source_coordinates = (int(organized_source[count][num][0]), int(organized_source[count][num][1]))
                gt_coordinates = (int(organized_gt[count][num][0]), int(organized_gt[count][num][1]))
                # print(center_coordinates)
                frame = cv2.circle(frame, center_coordinates, 5, (0, 255, 0), 2)  # green color
                frame = cv2.circle(frame, source_coordinates, 5, (0, 0, 255), 2)  # red color
                frame = cv2.circle(frame, gt_coordinates, 5, (252, 119, 3), 2)  # blue color

            # cv2.imshow('gt', frame)
            result.write(frame)
            # cv2.imwrite("frame%d.jpg" % count, frame)
        else:
            break
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    print("Video has been saved")
    cv2.destroyAllWindows()  # destroy all opened windows


def main():
    global openpose_
    openpose = False
    MODEL_DIR = "./checkpoints/hum36m/"
    model_name = 'Attention'

    MOTIONS = ['Walking', 'WalkDog', 'WalkTogether', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
               'Posing', 'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'Directions']
    keypoints_label = ['center hip', 'right hip', 'right knee', 'right ankle', 'left hip',
                       'left knee', 'left ankle', 'center body',
                       'neck', 'head', 'left shoulder', 'left elbow',
                       'left hand', 'right shoulder', 'right elbow', 'right hand']

    # graph parameter
    keypoint_loss = []
    ms = ['80', '160', '320', '400', '560', '720', '880', '1000']
    mpjpe = [[], [], [], [], [], [], [], []]
    mpjve = [[], [], [], [], [], [], [], []]
    all_loss_motion = []
    count = 0

    for motion in MOTIONS:
        print("motion = ", motion)
        if openpose == True:
            get_video = "./data/OpenPose/Video/" + motion + ".mp4"
            openpose_ = "./data/OpenPose/csv/" + motion + ".csv"  # /Video/repair
        else:
            get_video = "./Video/" + motion + "/S5/WalkDog.55011271.mp4"

        model_dirname = MODEL_DIR + model_name + motion + '.pt'
        model = torch.load(model_dirname, map_location=device)

        print("--------------------LOADING DATA----------------------")
        human36m = Human36M(folder='./data/', cam_angle=[1], motion=motion)
        real_test_data_source, real_test_data_target = human36m.get_data('test')

        if openpose:
            test = data_openpose(motion, openpose_)
            test_data_source, test_data_target = sliding_window(test)
            """
            Prediction using OpenPose data
            """
            pred, test_loss, source, keypoint, gt = predict(model, test_data_source, test_data_target,
                                                            real_test_data_source, real_test_data_target,
                                                            devices=device)
        else:
            pred, test_loss, source, keypoint, gt = predict(model, real_test_data_source, real_test_data_target,
                                                            devices=device)
        """
        Visualization
        """
        plot_humanposeGraph(gt[7], pred[7], motion, count)
        to_video_slide(pred[7], source, gt[7], get_video, motion)

        for msec in range(len(ms)):
            mpjpe_loss, mpjve_loss, mse_loss, all_loss = calc_testloss(pred[msec], gt[msec])
            print('msec = ', ms[msec], ', mpjpe_loss ', motion, ' = ', mpjpe_loss, ', mpjvee_loss ', motion, ' = ',
                  mpjve_loss, ', mse_loss ', motion, ' = ', mse_loss)
            mpjpe[msec].append(mpjpe_loss)
            mpjve[msec].append(mpjve_loss)

            if msec == 7:
                all_loss_motion.append(all_loss)

        # save keypoint loss
        keypoint_loss.append(keypoint)
        count += 1
    print('average mpjpe = ', np.mean(mpjpe, axis=1), ', average mpjve = ', np.mean(mpjve, axis=1))

    # seaborn heatmap graph for keypoints every motion
    accuracy_plot(keypoint_loss, keypoints_label, MOTIONS, model_name)


if __name__ == '__main__':
    main()
