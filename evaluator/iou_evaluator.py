# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import os
import re
from utilities import data_utils
import os.path as osp
import numpy as np
import pandas as pd
import argparse

''' 
DataHack evaluator based on iou metric. Use evaluate_dataset() to evaluate your predictions.
The evaluator assumes two top folders (one for ground truth and the other for predictions) that contain multiple 
subfolders with CSV files in them. The CSV files are of the same format as input data.
The same code will evaluate your final submissions.
reference https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task
'''


def evaluate_frame(gt_labels, pred_labels):
    assert np.all(np.isin(pred_labels, (0, 1))), \
        'Invalid values: pred labels value should be either 0 or 1, got {}'.format(set(pred_labels))

    correct_predictions = gt_labels == pred_labels
    positive_predictions = pred_labels == 1

    # correct, positive prediction -> True positive
    tp = np.sum(correct_predictions & positive_predictions)

    # incorrect, negative prediction (using De Morgan's law) -> False negative
    fn = np.sum(np.logical_not(correct_predictions | positive_predictions))

    # incorrect, positive prediction -> False positive
    fp = np.sum(np.logical_not(correct_predictions) & positive_predictions)

    return tp, fn, fp


def evaluate_folder(gt_folder, pred_folder, validmode=False):
    # get all ground truth files
    gt_indices = [data_utils.filename_to_frame(file)
                  for file in data_utils.find_sorted_files(gt_folder, 'labels')]
    print('evaluating folder {} with {} frames...'.format(osp.basename(gt_folder), len(gt_indices)))
    agg_tp = 0
    agg_fn = 0
    agg_fp = 0
    for frame_idx in gt_indices:
        pred_file_path = data_utils.frame_to_filename(pred_folder, frame_idx, 'labels')
        gt_labels = data_utils.read_data(gt_folder, frame_idx, 'labels')
        if not osp.isfile(pred_file_path):
            if validmode:
                continue
            else:
                print("No matching prediction file for frame {} in {}, filling zero labels".format(frame_idx, gt_folder))
                pred_labels = np.zeros_like(gt_labels)
        else:
            pred_labels = data_utils.read_data(pred_folder, frame_idx, 'labels')

        if len(gt_labels) != len(pred_labels):
            raise ValueError('GT point count ({}) does not match predicion point count ({}) in file: {}'
                             .format(len(gt_labels), len(pred_labels), frame_idx))

        tp, fn, fp = evaluate_frame(gt_labels, pred_labels)
        agg_tp += tp
        agg_fn += fn
        agg_fp += fp
    return agg_tp, agg_fn, agg_fp


def evaluate_dataset(top_folder_gt, top_folder_pred, validmode=False):
    # find all folders in top_folder_gt
    folders = [e for e in os.listdir(top_folder_gt) if osp.isdir(osp.join(top_folder_gt, e))]
    if len(folders) == 0:
        raise ValueError("No folders inside ground truth folder {}".format(top_folder_gt))

    if validmode:
        # keep only folders that have matches in predictions folder
        folders = [folder for folder in folders if osp.isdir(osp.join(top_folder_pred, folder))]
    else:
        # verify all folders in gt have corresponding folders in top_folder_pred
        for folder in folders:
            if not osp.isdir(osp.join(top_folder_pred, folder)):
                raise ValueError("No matching prediction folder for {}".format(folder))

    agg_tp = 0
    agg_fp = 0
    agg_fn = 0
    # evaluate and aggregate data from all folders
    for folder in folders:
        tp, fn, fp = evaluate_folder(osp.join(top_folder_gt, folder), osp.join(top_folder_pred, folder), validmode)
        agg_tp += tp
        agg_fn += fn
        agg_fp += fp
        # calculate IoU
        if agg_tp + agg_fp + agg_fn == 0:
            iou = 1.
        else:
            iou = agg_tp / float(agg_tp + agg_fp + agg_fn)
        print("IoU: {:.2f}%".format(iou*100))

    return iou


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth', help='ground truth folder', default=None)
    parser.add_argument('--predictions', help='predictions folder', default=None)
    parser.add_argument('--validmode', help='only evaluate frames that exist', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    assert osp.isdir(args.groundtruth), "--groundtruth should be a folder"
    assert osp.isdir(args.predictions), "--predictions should be a folder"
    iou = evaluate_dataset(args.groundtruth, args.predictions, args.validmode)
    print("IoU: {}".format(iou))