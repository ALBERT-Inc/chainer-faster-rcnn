#!/usr/bin/env python
# -*- coding: utf-8 -*-

# taken and modified from github.com/mitmul/chainer-fast-rcnn/scripts/forward.py

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'lib')
import time
import argparse
import cv2 as cv
import numpy as np
import cPickle as pickle
from vgg16_faster_rcnn import VGG16_faster_rcnn
from chainer import cuda
from nms.cpu_nms import cpu_nms

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)


def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale


def make_bboxes2(proposals, im_scale, min_size, dedup_boxes=1. / 16):   # proposals are scaled
    rects = [[0, d[0], d[1], d[2], d[3]] for d in proposals]
    rects = np.asarray(rects, dtype=np.float32)

    # bbox pre-processing
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    hashes = np.round(rects * dedup_boxes).dot(v)    # unique hash val
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    rects = rects[index, :]

    return rects


def draw_result(out, im_scale, clss, bbox, rects, nms_thresh, conf, draw=True):
    if draw:
        out = cv.resize(out, None, None, fx=im_scale, fy=im_scale, 
                        interpolation=cv.INTER_LINEAR)
    else:
        out = 0
    mdets = np.zeros((0, 6))   # [l, t, b, r, prob, class]
    for cls_id in range(1, 21):
        _cls = clss[:, cls_id][:, np.newaxis]   # classification score
        _bbx = bbox[:, cls_id * 4: (cls_id + 1) * 4]
        dets = np.hstack((_bbx, _cls))
        keep = cpu_nms(dets, nms_thresh)
        dets = dets[keep, :]
        orig_rects = cuda.cupy.asnumpy(rects)[keep, 1:]

        inds = np.where(dets[:, -1] >= conf)[0]
        for i in inds:
            _bbox = dets[i, :4]
            x1, y1, x2, y2 = orig_rects[i]
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + 0.5 * width
            center_y = y1 + 0.5 * height

            dx, dy, dw, dh = _bbox
            _center_x = dx * width + center_x
            _center_y = dy * height + center_y
            _width = np.exp(dw) * width
            _height = np.exp(dh) * height

            x1 = _center_x - 0.5 * _width
            y1 = _center_y - 0.5 * _height
            x2 = _center_x + 0.5 * _width
            y2 = _center_y + 0.5 * _height
            mdets = np.vstack((mdets, np.hstack(([x1, y1, x2, y2, dets[i, -1], cls_id]))))
            
            if draw:
                cv.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                             (0, 0, 255), 2)
                ret, baseline = cv.getTextSize(CLASSES[cls_id],
                                               cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
                cv.rectangle(out, (int(x1), int(y2) - ret[1] - baseline),
                             (int(x1) + ret[0], int(y2)), (0, 0, 255), -1)
                cv.putText(out, CLASSES[cls_id], (int(x1), int(y2) - baseline),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1,
                           )

    return out, mdets


import pylab as plt
import cv2
from rpn.make_proposals import make_proposals
from rpn.make_proposals import _num_anchors
import chainer.functions as F
from chainer import serializers

vgg = VGG16_faster_rcnn()
serializers.load_hdf5('./faster_rcnn_models/VGG16_faster_rcnn.model', vgg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', type=str, default='sample.jpg')
    parser.add_argument('--out_fn', type=str, default='result.jpg')
    parser.add_argument('--min_size', type=int, default=500)
    parser.add_argument('--nms_thresh', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu) 
        xp = cuda.cupy
        vgg.to_gpu()
    else:
        xp = np
    
    global orig_image, im_scale
    orig_image = cv.imread(args.img_fn)
    img, im_scale = img_preprocessing(orig_image, PIXEL_MEANS)
    img = xp.asarray(img)

    conv5_3, rpn_bbox, rpn_cls = vgg.forward(img[xp.newaxis, :, :, :], train=False)

    bbox_deltas = cuda.cupy.asnumpy(rpn_bbox.data)
    scores = cuda.cupy.asnumpy(rpn_cls.data)
    scores = (scores[:, _num_anchors:] - scores[:, :_num_anchors])
    global proposals
    proposals, _ = make_proposals(bbox_deltas, scores, im_scale)

    orig_rects = make_bboxes2(proposals, im_scale, min_size=args.min_size)
    rects = xp.asarray(orig_rects)
    cls_score, bbox_pred = vgg.forward_rcnn(conv5_3, rects, train=False)
    cls_score = F.softmax(cls_score)
    conv5_3 = 0
    
    clss = cuda.cupy.asnumpy(cls_score.data)
    bbox = cuda.cupy.asnumpy(bbox_pred.data)
    result, m_detects = draw_result(orig_image, im_scale, clss, bbox, orig_rects,
                         args.nms_thresh, args.conf)
    cv2.imwrite(args.out_fn, result)
    plt.imshow(result[:,:,::-1])


