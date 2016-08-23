# Chainer Faster R-CNN
Implementation of detection facility of Faster R-CNN in Chainer.

It is based on both https://github.com/rbgirshick/py-faster-rcnn and https://github.com/mitmul/chainer-fast-rcnn.

## Requirements
OpenCV 3.0 with python bindings  
Chainer 1.6 

## Installation
1. Clone the Faster R-CNN repository.  
    `hoge`

1. Build the Cython modules(\$FRCN_ROOT is the directory that you cloned Faster R-CNN into)  
`cd $FRCN_ROOT/lib`  
`make`

1. Download pre-trained Faster R-CNN model  
`cd $FRCN_ROOT`    
`./scripts/fetch_faster_rcnn_models.sh`

1. Convert model  
`$ python scripts/convert_vgg16.py`

## Demo
`$ python scripts/faster-rcnn.py --img_fn demo/000456.jpg --out_fn result.jpg`
Use `--gpu -1` if computing without GPU.
