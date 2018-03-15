cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/coco/DES512x512/solver_test_dev.prototxt" \
--weights="models/VGGNet/coco/DES512x512/VGG_coco_SSD_512x512_iter_360000.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/coco/DES512x512/DES512x512_test_dev.log
