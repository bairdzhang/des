cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/coco/DES300x300/solver_test_dev.prototxt" \
--weights="models/VGGNet/coco/DES300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/coco/DES300x300/DES300x300_test_dev.log
