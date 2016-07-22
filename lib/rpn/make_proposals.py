import numpy as np
import sys

sys.path.insert(0, '../../lib')
from nms.cpu_nms import cpu_nms as nms
from fast_rcnn.config import cfg

_anchors = np.array([
        [ -83.,  -39.,  100.,   56.],
        [-175.,  -87.,  192.,  104.],
        [-359., -183.,  376.,  200.],
        [ -55.,  -55.,   72.,   72.],
        [-119., -119.,  136.,  136.],
        [-247., -247.,  264.,  264.],
        [ -35.,  -79.,   52.,   96.],
        [ -79., -167.,   96.,  184.],
        [-167., -343.,  184.,  360.]])

# Translating from prediction to proposals
# Code taken and modified from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py
# 2016/03 Modification by Tsuguo Mogami

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
#from nms import nms

_feat_stride = 16
_num_anchors = len(_anchors)

def make_proposals(bbox_deltas, scores, im_scale, train=False):
    cfg_key = 'TRAIN' if train else 'TEST'
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N    # 12000
    post_nms_topN= cfg[cfg_key].RPN_POST_NMS_TOP_N   # 2000
    nms_thresh   = cfg[cfg_key].RPN_NMS_THRESH   # 0.7
    min_size     = cfg[cfg_key].RPN_MIN_SIZE
    
    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    #proposals = clip_boxes(proposals, im_info[:2])	# todo: revert

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_scale) # TM: im_info[2] -> im_scale
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    return proposals, scores

