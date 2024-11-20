import os
import os.path as osp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_bbox(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()

    lines = [[float(i) for i in line.split(" ")] for line in lines]

    return torch.tensor(lines)

def compute_iou(ground_truth, predictions):
    gt_cls, gt_bbox = int(ground_truth[0]), ground_truth[1:]
    pred_cls, pred_bboxes = predictions[:, 0].long(), predictions[:, 1:]

    x0_gt, y0_gt, x1_gt, y1_gt = gt_bbox
    x0, y0, x1, y1 = pred_bboxes.chunk(4, dim=1)

    interset_x0 = x0.clamp(min=x0_gt)
    interset_y0 = y0.clamp(min=y0_gt)
    interset_x1 = x1.clamp(max=x1_gt)
    interset_y1 = y1.clamp(max=y1_gt)
    delta_x = (interset_x1 - interset_x0).clamp(min=0)
    delta_y = (interset_y1 - interset_y0).clamp(min=0)
    interset_area = delta_x * delta_y

    overset_x0 = x0.clamp(max=x0_gt)
    overset_y0 = y0.clamp(max=y0_gt)
    overset_x1 = x1.clamp(min=x1_gt)
    overset_y1 = y1.clamp(min=y1_gt)
    delta_x = (overset_x1 - overset_x0).clamp(min=0)
    delta_y = (overset_y1 - overset_y0).clamp(min=0)
    overset_area = delta_x * delta_y

    iou = interset_area / overset_area
    iou[pred_cls != gt_cls] = -1

    return iou.view(-1)

def match(ground_truth, predictions, iou_thres=0.5):

    matched_gt = [0] * len(ground_truth)
    matched_pred = [0] * len(predictions)
    for i, gt in enumerate(ground_truth):
        iou = compute_iou(gt, predictions)
        match_index = iou.argmax()
        if iou[match_index] < iou_thres:
            continue

        matched_gt[i] = 1
        matched_pred[match_index] = 1

    return torch.tensor(matched_gt), torch.tensor(matched_pred)

def compute_precision_recall(det_root, iou_thres):
    ground_truth = load_bbox(osp.join(det_root, "exp/labels/original_image.txt"))

    all_predictions = []
    while True:
        try:
            all_predictions.append(load_bbox(osp.join(det_root, "exp/labels/%06d.txt"%len(all_predictions))))
        except:
            break

    precision, recall = [], []
    for predictions in tqdm(all_predictions):
        matched_gt, matched_pred = match(ground_truth, predictions, iou_thres)
        prec = matched_pred.float().mean()
        rec = matched_gt.float().mean()
        precision.append(prec)
        recall.append(rec)

    precision = torch.tensor(precision)
    recall = torch.tensor(recall)

    return precision.unsqueeze(0), recall.unsqueeze(0)

def compute_precision_recall_per_object(det_root, iou_thres):
    ground_truth = load_bbox(osp.join(det_root, "exp/labels/original_image.txt"))

    all_predictions = []
    while True:
        try:
            all_predictions.append(load_bbox(osp.join(det_root, "exp/labels/%06d.txt"%len(all_predictions))))
        except:
            break

    all_precision, all_recall = [], []
    for gt in ground_truth:
        precision, recall = [], []
        for predictions in all_predictions:
            matched_gt, matched_pred = match(gt.unsqueeze(0), predictions, iou_thres)
            prec = matched_pred.float().mean()
            rec = matched_gt.float().mean()
            precision.append(prec)
            recall.append(rec)

        precision = torch.tensor(precision)
        recall = torch.tensor(recall)

        all_precision.append(precision)
        all_recall.append(recall)

    all_precision = torch.stack(all_precision, 0)
    all_recall = torch.stack(all_recall, 0)

    return all_precision, all_recall
    
def main(det_root, mask_path, shapley_value_fn, per_object=False, iou_thres=0.5):
    keep_features = torch.from_numpy(np.load(mask_path))
    if per_object:
        precision, recall = compute_precision_recall_per_object(det_root, iou_thres)
    else:
        precision, recall = compute_precision_recall(det_root, iou_thres)

    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    N, H, W = keep_features.shape

    shapley_values = []

    for i in range(len(f1_score)):
        shapley_value = shapley_value_fn(keep_features, f1_score[i])
        shapley_values.append(shapley_value)
        # cv2.imwrite("tmp_%d.jpg"%i, (shapley_value * 255).numpy().astype(np.uint8))

    shapley_values = torch.stack(shapley_values)

    return shapley_values.numpy()


if __name__ == "__main__":
    main("./shap/bus/detect_results", "./shap/bus/masked_images/masks.npy")
