import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import config
import resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import roi_helpers
import data_generators
from sklearn.metrics import average_precision_score


def get_map(pred,gt,f):
    T={}
    P={}
    fx,fy=f
    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array(s['prob'] for s in pred)
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class]=[]
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = data_generators.iou((pred_x1,pred_y1,pred_x2,pred_y2),(gt_x1,gt_y1,gt_x2,gt_y2))
            if iou >=0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue
        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []
            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)
    return T, P


# Don't recurse to much
sys.setrecursionlimit(40000)

parser = OptionParser()
parser.add_option('-p','--path',dest='test_path',help='Path to test data')
parser.add_option('-n','--num_rois',dest='num_rois',
                  help='NUmber of ROIs per iteration.Higher means more memory use. ', default=32)
parser.add_option('--config_filename',dest='config_filename',
                  help='Location to read the metadata related to the training (generated when training)')









