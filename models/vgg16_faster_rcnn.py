# taken and modified from https://github.com/mitmul/chainer-fast-rcnn/blob/master/models/VGG.py
import sys
sys.path.insert(0, 'functions')
from chainer import Variable, FunctionSet
import chainer.functions as F
from roi_pooling_2d import roi_pooling_2d

class VGG16_faster_rcnn(FunctionSet):
    def __init__(self):
        super(VGG16_faster_rcnn, self).__init__(
            conv1_1 = F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2 = F.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1 = F.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2 = F.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = F.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2 = F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3 = F.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1 = F.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2 = F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3 = F.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1 = F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2 = F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3 = F.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6 = F.Linear(512*7*7, 4096),
            fc7 = F.Linear(4096, 4096),
            cls_score = F.Linear(4096, 21),
            bbox_pred = F.Linear(4096, 84),
            rpn_conv_3x3 = F.Convolution2D(512, 512, 3, stride=1, pad=1),
            rpn_cls_score = F.Convolution2D(512, 18, 1, stride=1, pad=0),
            rpn_bbox_pred = F.Convolution2D(512, 36, 1, stride=1, pad=0)
        )

    def forward(self, x_data, train=True):
        x = Variable(x_data, volatile=True)

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        conv5_3 = F.relu(self.conv5_3(h))
        conv5_3 = Variable(conv5_3.data, volatile = not train)   # cut backprop here
        
        rpn_conv = F.relu(self.rpn_conv_3x3(conv5_3))
        rpn_bbox = self.rpn_bbox_pred(rpn_conv)
        rpn_cls = self.rpn_cls_score(rpn_conv)
        return conv5_3, rpn_bbox, rpn_cls
    
    def forward_rcnn(self, conv5_3, rois, train=False):
        rois = Variable(rois, volatile=not train)
        h = roi_pooling_2d(conv5_3, rois)

        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        cls_score = (self.cls_score(h))
        bbox_pred = self.bbox_pred(h)

        return cls_score, bbox_pred
