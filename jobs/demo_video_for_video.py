#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


import matplotlib
from matplotlib.pyplot import plot,savefig
import cv2.cv as cv

CLASSES = ('__background__','car','person')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg_m': ('VGG_CNN_M_1024',
                   'VGG_CNN_M_1024_faster_rcnn_final.caffemodel')}


def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    index=1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        #im = im[:, :, (2, 1, 0)]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0 and index==len(CLASSES[1:]):
            #cv2.imwrite(path,im)
            video.write(im)
            return
        elif len(inds) == 0 and index<len(CLASSES[1:]):
            index+=1
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            x = bbox[0]
            y = bbox[1]
            rect_start = (x,y)
            x1 = bbox[2]
            y1 = bbox[3]
            rect_end = (x1,y1)
            color0=(100,100,100)
            color1=(255,0,0)


            xx1 = bbox[0]
            yy1= int(bbox[1]-10)
            point_start = (xx1,yy1)
            xx2 = bbox[0]+(bbox[2]-bbox[0])*score
            yy2= int(bbox[1]-2)
            point_end = (xx2,yy2)
            color2=(0,0,225)
            color3=(0,255,0)
            if cls_ind==1:
                cv2.rectangle(im, rect_start, rect_end, color1, 2)
            elif cls_ind==2:
                cv2.rectangle(im, rect_start, rect_end, color3, 2)
            elif cls_ind==3:
                cv2.rectangle(im, rect_start, rect_end, color0, 2)
            cv2.rectangle(im, point_start, point_end, color2, -1)
    cv2.namedWindow("Image")
    res=cv2.resize(im,(1080,608),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Image", res)
    cv2.waitKey (1)
    #cv2.imwrite(path,im)
    #video.write(im)




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg_m')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    #im = 128 * np.ones((1280, 960, 3), dtype=np.uint8)
    #for i in xrange(2):
    #   _, _= im_detect(net, im)

    #dictionary=['20121117_142852', '20121117_150315', '20121117_153526', '20121128_092059', '20121130_095032', '20130110_135753', '20130110_140950', '20130110_142518', '20130123_094123', '20130123_112228', '20130123_132342', '20130123_143631', '20130129_133540', '20130311_112935', '20130311_115905', '20130314_102842', '20130314_144414', '20130319_121354']
    dir_name='1.avi'

    dir_root=os.path.join(cfg.ROOT_DIR, 'data', 'demo',dir_name)

    videoCapture = cv2.VideoCapture(dir_root)
    fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
   # fps=25
    size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    success, frame = videoCapture.read()


    #cv2.cv.CV_FOURCC('I','4','2','0') avi
    #cv2.cv.CV_FOURCC('P','I','M','1') avi
    #cv2.cv.CV_FOURCC('M','J','P','G') avi
    #cv2.cv.CV_FOURCC('T','H','E','O') ogv
    #cv2.cv.CV_FOURCC('F','L','V','1') flv
    video=cv2.VideoWriter(dir_name, cv2.cv.CV_FOURCC('M','J','P','G'), int(fps),size)
    if not video:
        print "Error in creating video writer"
    while success :
        demo(net,frame)
        success, frame = videoCapture.read()
    video.release()
