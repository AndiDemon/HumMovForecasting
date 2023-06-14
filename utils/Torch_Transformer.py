import numpy as np
import torch

"""Loss Function"""


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def mpjpe_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe_torch(predicted, target, with_sRt=False, full_torch=False, with_aligned=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = torch.mean(target, dim=1, keepdim=True)
    muY = torch.mean(predicted, dim=1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY
    X0[X0 ** 2 < 1e-6] = 1e-3

    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(1, 2), keepdim=True))

    normX[normX < 1e-3] = 1e-3

    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(1, 2), Y0)
    if full_torch:
        U, s, V = batch_svd(H)
    else:
        U, s, Vt = np.linalg.svd(H.cpu().numpy())
        V = torch.from_numpy(Vt.transpose(0, 2, 1)).cuda()
        U = torch.from_numpy(U).cuda()
        s = torch.from_numpy(s).cuda()

    R = torch.matmul(V, U.transpose(2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = torch.sign(torch.unsqueeze(torch.det(R[0]), 0))
    V[:, :, -1] *= sign_detR.unsqueeze(0)
    s[:, -1] *= sign_detR.flatten()
    R = torch.matmul(V, U.transpose(2, 1))  # Rotation

    tr = torch.unsqueeze(torch.sum(s, dim=1, keepdim=True), 2)

    a = tr * normX / normY  # Scale
    t = muX - a * torch.matmul(muY, R)  # Translation

    if (a != a).sum() > 0:
        print('NaN Error!!')
        print('UsV:', U, s, V)
        print('aRt:', a, R, t)
    a[a != a] = 1.
    R[R != R] = 0.
    t[t != t] = 0.
    # Perform rigid transformation on the input
    predicted_aligned = a * torch.matmul(predicted, R) + t
    if with_sRt:
        return torch.sqrt(((predicted_aligned - target) ** 2).sum(-1)).mean(), (
            a, R, t)  # torch.mean(torch.norm(predicted_aligned - target, dim=len(target.shape)-1))
    if with_aligned:
        return torch.sqrt(((predicted_aligned - target) ** 2).sum(-1)).mean(), predicted_aligned
    # Return MPJPE
    return torch.sqrt(((predicted_aligned - target) ** 2).sum(
        -1)).mean()  # torch.mean(torch.norm(predicted_aligned - target, dim=len(target.shape)-1))#,(a,R,t),predicted_aligned


def batch_svd(H):
    num = H.shape[0]
    U_batch, s_batch, V_batch = [], [], []
    for i in range(num):
        U, s, V = H[i].svd(some=False)
        U_batch.append(U.unsqueeze(0))
        s_batch.append(s.unsqueeze(0))
        V_batch.append(V.unsqueeze(0))
    return torch.cat(U_batch, 0), torch.cat(s_batch, 0), torch.cat(V_batch, 0)


def p_mpjpe(target, predicted, with_sRt=False, full_torch=False, with_aligned=False, each_separate=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= (normX + 1e-6)
    Y0 /= (normY + 1e-6)

    H = np.matmul(X0.transpose(0, 2, 1), Y0).astype(np.float16).astype(np.float64)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    if each_separate:
        return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)

    error = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))
    if with_sRt and not with_aligned:
        return error, (a, R, t)
    if with_aligned:
        return error, (a, R, t), predicted_aligned
    # Return MPJPE
    return error


def n_mpjpe(target, predicted):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    #
    # print("prediction = ", predicted.shape)
    # print("velocity = ", velocity_predicted.shape)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))


def MPJLE(predicted, gt, tolerance):
    """"
        predicted   = (frame, keypoints, 2*V) prediction results in motions respectively
        gt          = (frame, keypoints, 2*V) ground truth in motions respectively
        tolerance   = [interval start, interval end]
        output -> error = percentage of the correct keypoint in interval of tolerance
    """
    error = []
    for threshold in range(tolerance[0], tolerance[1]):
        e_thres = []
        for frame in range(gt.shape[0]):
            loss = np.linalg.norm(np.array(predicted[frame]) - np.array(gt[frame]))
            if loss <= threshold:
                e_thres.append(0)
            else:
                e_thres.append(1)
        error.append(np.mean(e_thres))
    return error


def IOU(predicted, gt, threshold=''):
    iou_all = []
    mAP = []
    for frame in range(predicted.shape[0]):
        min_x_pred, min_y_pred = np.amin(predicted[frame], axis=0)
        max_x_pred, max_y_pred = np.amax(predicted[frame], axis=0)

        min_x_gt, min_y_gt = np.amin(gt[frame], axis=0)
        max_x_gt, max_y_gt = np.amax(gt[frame], axis=0)

        # determine the xy coordinate of the intersection
        xA = max(min_x_pred, min_x_gt)
        yA = max(min_y_pred, min_y_gt)
        xB = min(max_x_pred, max_x_gt)
        yB = min(max_y_pred, max_y_gt)

        # compute intersection area rectangle
        area_intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute area of prediction and ground truth rectangle
        boxAArea = (max_x_pred - min_x_pred + 1) * (max_y_pred - min_y_pred + 1)
        boxBArea = (max_x_gt - min_x_gt + 1) * (max_y_gt - min_y_gt + 1)

        # compute intersection over union
        iou = area_intersection / float(boxAArea + boxBArea - area_intersection)
        iou_all.append(iou)

        # print('iou = ', iou)

    return np.mean(np.array(iou_all))


def rmse_ex(predictions, targets):
    """

    Args:
        predictions: prediction results
        targets: expected target from ground truth

    Returns: MPJPE

    """
    return np.sqrt(((predictions - targets) ** 2).mean())
