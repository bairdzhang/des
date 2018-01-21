cd /workspace/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES300x300/solver.prototxt" \
--weights="../models/VOC300TRAINED_4e04e3.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/VOC0712/DES300x300/DES300x300.log
