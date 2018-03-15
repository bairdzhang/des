cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES300x300withCOCO_12/solver12.prototxt" \
--weights="models/VGGNet/VOC0712/DES300x300withCOCO_12/VGG_VOC0712_SSD_300x300_ft_iter_160000.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/VGGNet/VOC0712/DES300x300withCOCO_12/DES300x300withCOCO_12_test.log
