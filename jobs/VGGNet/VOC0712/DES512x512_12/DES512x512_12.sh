cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES512x512_12/solver.prototxt" \
--weights="/workspace/models/VOC512TRAINED_02db43.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/VOC0712/DES512x512_12/DES512x512_12.log
