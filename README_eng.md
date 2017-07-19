# SSD detection system for KITTI

1. Data Preparation

A tool for converting KITTI-format labels to VOC-xml files is provided: data/KITTI/KITTI_xml.py.

Unpack KITTI training data to data/KITTI, images are under data/KITTI/image_2/, annotation labels under data/KITTI/label_2/.

Currently this project can detect 3 classes: Pedestrian, Car, Cyclist.
You can add other new classes in KITTI_xml.py, *class_ind*.

KITTI_xml.py will generate train.txt under data/KITTI. train.txt is used to store image path and label path of coresponding xml file. test_name_size.txt is used to store detected image and its width/height.

You can find class names and index in labelmap_KITTI.prototxt.

After that you can execute *create_data.sh* under root directory (with some path been modified appropriately),then you can generate *KITTI_train_lmdb* under data/KITTI. Same as for KITTI_test_lmdb.

(*para_show.sh* is for ease of displaying parameters in create_data.sh.)

create_data.sh calls scripts/create_annoset.py to create LMDBs (create_annoset_debug.py is for debug purpose).

After creating LMDBs, create symbolic links to examples/KITTI (be careful about modifying those absolute paths).

2. Train and test

Run examples/ssd/ssd_KITTI.py which will create VGGNet-KITTI prototxt files to directory models/, and validating/testing scripts to jobs/VGGNet/.

You need to put VGG_ILSVRC_16_layers_fc_reduced.caffemodel under this same directory.

SSD_600x150/VGG_KITTI_SSD_600x150.sh is used for training and validating, SSD_600x150_test/VGG_KITTI_SSD_600x150.sh is used for testing.

Note, the major difference between training solver and testing solver is that the latter has a max_iter = 0.

After finishing training, you can run jobs/VGGNet/KITTI/SSD_600x150_webcam/VGG_KITTI_SSD_600x150.sh to 
utilize USB camera to do real-time detectin. 

Here I modified *VideoData* in caffe.proto in order to support detection on local video file.
Detail steps will be updated on my blogs.
(you will need to modify some *.py files under examples/ssd)

All source codes are implemented using C++, no Python interface functions are provided.

I also implemented *detection.py* for detection on images, *detection_video.py* for converting images detection results to videos, and *demo_video_for_video.py* for detection on local video files.

Note: all absolute/hard-coded paths need to be modified.

If you have any problems or find some bugs, please contact with manutdzou@126.com and feel free. Thank you again for your observation.