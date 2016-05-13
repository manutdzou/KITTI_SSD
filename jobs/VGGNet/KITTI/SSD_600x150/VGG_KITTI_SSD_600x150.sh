cd /home/bsl/Debug/ssd_caffe
./build/tools/caffe train \
--solver="/home/bsl/Debug/ssd_caffe/models/VGGNet/KITTI/SSD_600x150/solver.prototxt" \
--weights="/home/bsl/Debug/ssd_caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
---iterations=7481 \
--gpu 0 2>&1 | tee /home/bsl/Debug/ssd_caffe/jobs/VGGNet/KITTI/SSD_600x150/VGG_KITTI_SSD_600x150.log
