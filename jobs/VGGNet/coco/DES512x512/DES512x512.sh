cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/coco/DES512x512/solver.prototxt" \
--weights="/workspace/models/COCO512RESET_b67be6.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/coco/DES512x512/DES512x512.log
