cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES300x300/solver_eval.prototxt" \
--weights="models/VGGNet/VOC0712/DES300x300/DES300x300_iter_120000.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/VOC0712/DES300x300/DES300x300_eval.log
