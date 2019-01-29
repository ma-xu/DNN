import numpy as np
import math
import data_generators
import copy


def calc_iou(R,img_data,C, class_mapping):
    bboxes = img_data['bboxes']
    (width,height) = (img_data['width'],img_data['height'])
    #get image dimensions for resizing
    (resized_width,resized_height) = data_generators.get_new_img_size(width,height,C.im_size)

    gta = np.zeros((len(bboxes),4))

    for bbox_num,bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride ))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride ))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] #for debugging only

    for ix in range(R.shape[0]):
        (x1,y1,x2,y2) = R[ix,:]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou =0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = data_generators.iou([gta[bbox_num, 0],gta[bbox_num, 2],gta[bbox_num, 1],gta[bbox_num, 3]],
                                           [x1,y1,x2,y2])
            if curr_iou>best_iou:
                best_iou=curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
            continue
        else:
            w = x2-x1
            h = y2-y1
            x_roi.append([x1,y1,w,h])
            IoUs.append(best_iou)
            if C.classifier_min_overlap <=best_iou < C.classifier_max_overlap:
                cls_name = 'bg'
            elif C.classifier_max_overlap <=best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox,0]+gta[best_bbox,1])/2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0
                cx = x1+w/2.0
                cy = y1+h/2.0

                tx = (cxg-cx)/float(w)
                ty = (cyg-cy)/float(h)
                tw = np.log((gta[best_bbox,1]-gta[best_bbox,0])/float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi={}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping)*[0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        









