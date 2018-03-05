cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES300x300_12/solver.prototxt" \
--weights="../../models/VOC300TRAINED_4e04e3.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/VOC0712/DES300x300_12/DES300x300_12.log
