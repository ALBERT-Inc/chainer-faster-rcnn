import chainer
from chainer.functions import caffe
from chainer import serializers

print 'Loading caffemodel...'
func = chainer.functions.caffe.CaffeFunction('./faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel')
print 'Converting...'
func.rpn_conv_3x3 = getattr(func, u'rpn_conv/3x3')
func._children.append('rpn_conv_3x3')
serializers.save_hdf5('./faster_rcnn_models/VGG16_faster_rcnn.model', func)
print 'Done.'
