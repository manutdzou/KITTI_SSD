cd /home/bsl/Debug/ssd_caffe
./build/tools/caffe test \
--model="/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150_webcam/test.prototxt" \
--weights="/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150_iter_60000.caffemodel" \
--iterations="536870911" \
--gpu 0
