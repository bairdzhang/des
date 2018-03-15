cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES300x300withCOCO/solver.prototxt" \
--weights="/workspace/models/DES300x300_coco_to_voc.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/VOC0712/DES300x300withCOCO/DES300x300withCOCO.log
