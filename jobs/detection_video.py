import sys
sys.path.append('/home/bsl/caffe/python/')
import caffe
import os
import numpy as np
import cv2
import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
caffe_root = "/home/bsl/Debug/ssd_caffe/"
if os.path.isfile(caffe_root + 'models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'CaffeNet not found'
model_def = caffe_root + 'models/VGGNet/KITTI/SSD_600x150/deploy_large.prototxt'
model_weights = caffe_root + 'models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
mu = np.array([104, 117, 123])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,3,270, 480)

test_image_path=caffe_root+'data/KITTI/training/data_object_image_2/testing/image_2'
color=[(255,0,0),(0,255,0),(0,0,255)]
visualize_threshold=0.6


dir_name='04041652_2624.MP4'
dir_root=os.path.join(caffe_root,dir_name)
videoCapture = cv2.VideoCapture(dir_root)
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
#fps=25
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))/2,int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))/2)
success, frame = videoCapture.read()


#cv2.cv.CV_FOURCC('I','4','2','0') avi
#cv2.cv.CV_FOURCC('P','I','M','1') avi
#cv2.cv.CV_FOURCC('M','J','P','G') avi
#cv2.cv.CV_FOURCC('T','H','E','O') ogv
#cv2.cv.CV_FOURCC('F','L','V','1') flv
video=cv2.VideoWriter(dir_name, cv2.cv.CV_FOURCC('M','J','P','G'), int(fps),size)

while success:
    timer=Timer()
    image=frame/255.
    #image = caffe.io.load_image(img_path)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    timer.tic()
    output = net.forward() #detectors 1*1*N*7 N*(image-id, label, confidence, xmin, ymin, ymax)
    timer.toc()
    shape=output['detection_out'].shape
    detectors=output['detection_out'].reshape(shape[2],shape[3])
    #visualize
    img=cv2.resize(frame,(size[1],size[0]))
    for i in xrange(detectors.shape[0]):
        if detectors[i][2]>=visualize_threshold:
            xmin=int(detectors[i][3]*size[1])
            ymin=int(detectors[i][4]*size[0])
            xmax=int(detectors[i][5]*size[1])
            ymax=int(detectors[i][6]*size[0])
            label=detectors[i][1]
            rect_start=(xmin,ymin)
            rect_end=(xmax,ymax)
            cv2.rectangle(img, rect_start, rect_end, color[int(label-1)], 2)
    cv2.imshow('image',img)
    cv2.waitKey(1)
    print ('Detection took {:.3f}s').format(timer.total_time)
    success, frame = videoCapture.read()
