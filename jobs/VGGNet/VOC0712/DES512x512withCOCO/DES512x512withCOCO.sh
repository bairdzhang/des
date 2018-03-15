cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES512x512withCOCO/solver.prototxt" \
--weights="/workspace/models/DES512x512_coco_to_voc.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/VOC0712/DES512x512withCOCO/DES512x512withCOCO.log
