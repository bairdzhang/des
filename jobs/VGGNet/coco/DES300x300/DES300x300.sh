cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/coco/DES300x300/solver.prototxt" \
--weights="/workspace/models/COCO300RESET_676753.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/coco/DES300x300/DES300x300.log
