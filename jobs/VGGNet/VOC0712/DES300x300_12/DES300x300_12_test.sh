cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES300x300_12/solver_test.prototxt" \
--weights="models/VGGNet/VOC0712/DES300x300_12/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/VOC0712/DES300x300_12/DES300x300_12_test.log
