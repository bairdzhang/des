cd /workspace/release/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/DES512x512withCOCO_12/solver12.prototxt" \
--weights="models/VGGNet/VOC0712/DES512x512withCOCO_12/VGG_VOC0712_SSD_512x512_ft_iter_160000.caffemodel" \
--gpu 0,1 2>&1 | tee jobs/VGGNet/VOC0712/DES512x512withCOCO_12/DES512x512withCOCO_12_test.log
