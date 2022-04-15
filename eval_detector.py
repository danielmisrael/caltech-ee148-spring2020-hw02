import os
import json
import numpy as np
import matplotlib.pyplot as plt

def get_area(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)
    assert area >= 0
    return area

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # tl_col, tl_row, br_col, br_row
    x1, y1, x2, y2 =  box_1[0], box_1[1], box_1[2], box_1[3]
    a1, b1, a2, b2 = box_2[0], box_2[1], box_2[2], box_2[3]


    assert x2 > x1 and y2 > y1 and a2 > a1 and b2 > b1

    tl_col = max(x1, a1)
    tl_row = max(y1, b1)
    br_col = min(x2, a2)
    br_row = min(y2, b2)

    if br_col < tl_col or br_row < tl_row:
        return 0

    intersection = get_area(tl_col, tl_row, br_col, br_row)
    union = get_area(x1, y1, x2, y2) + get_area(a1, b1, a2, b2) - intersection

    iou = intersection / union

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                conf = pred[j][4]
                # print("pred", pred[j])
                # print("conf", conf)
                # print("conf_thr", conf_thr)
                # print("iou", iou)
                if iou >= iou_thr and conf >= conf_thr:
                    TP += 1
                elif iou < iou_thr and conf >= conf_thr:
                    FP += 1
                elif iou >= iou_thr and conf < conf_thr:
                    FN += 1
    '''
    END YOUR CODE
    '''


    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

# print(preds_train)
confidence_thrs = np.sort(np.array([pred[4] for fname in preds_train for pred in preds_train[fname]],dtype=float).flatten()) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for iou_thr in [0.05, 0.25, 0.5, 0.75]:
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

    # Plot training set PR curves

    recall = tp_train / (tp_train + fn_train)
    precision = tp_train / (tp_train + fp_train)
    plt.plot(recall, precision)
    plt.title(f"PR Curve for IOU {iou_thr}")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

if done_tweaking:
    print('Code for plotting test set PR curves.')
