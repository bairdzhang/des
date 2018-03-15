#!/usr/bin/env python
import sys
sys.path.insert(0, './python')
import numpy as np
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
caffe.set_device(3)
caffe.set_mode_gpu()
coco_net = caffe.Net('./models/VGGNet/coco/DES300x300/train.prototxt',
                     './models/VGGNet/coco/DES300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel',
                     caffe.TRAIN)
voc_net = caffe.Net('./models/VGGNet/VOC0712/DES300x300/train.prototxt',
                    './models/VGGNet/VOC0712/DES300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel',
                    caffe.TRAIN)
map_file = './data/VOC0712/coco_voc_map.txt'
maps = np.loadtxt(map_file, str, delimiter=',')

coco_labelmap_file = './data/coco/labelmap_coco.prototxt'
file = open(coco_labelmap_file, 'r')
coco_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), coco_labelmap)

voc_labelmap_file = './data/VOC0712/labelmap_voc.prototxt'
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

for m in maps:
    [coco_label, voc_label, name] = m
    coco_name = get_labelname(coco_labelmap, int(coco_label))[0]
    voc_name = get_labelname(voc_labelmap, int(voc_label))[0]
    assert voc_name == name
    print('{}, {}'.format(coco_name, voc_name))

mbox_source_layers = ['conv4_3_norm', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
num_bboxes = [4, 6, 6, 6, 4, 4]
assert len(mbox_source_layers) == len(num_bboxes)
num_voc_classes = 21
num_coco_classes = 81

def sample_param(src_param, src_num_classes, dst_num_classes, num_bboxes, maps):
    src_shape = src_param.shape
    assert src_shape[0] == src_num_classes * num_bboxes
    if len(src_shape) == 4:
        dst_shape = (dst_num_classes * num_bboxes, src_shape[1], src_shape[2], src_shape[3])
    else:
        dst_shape = dst_num_classes * num_bboxes
    dst_param = np.zeros(dst_shape)
    for i in xrange(0, num_bboxes):
        for m in maps:
            [src_label, dst_label, name] = m
            src_idx = i * src_num_classes + int(src_label)
            dst_idx = i * dst_num_classes + int(dst_label)
            dst_param[dst_idx,] = src_param[src_idx,]
    return dst_param
    
for layer_name, param in coco_net.params.iteritems():
    if 'mbox' not in layer_name and layer_name != 'seg':
        for i in xrange(0, len(param)):
            voc_net.params[layer_name][i].data.flat = coco_net.params[layer_name][i].data.flat

for i in xrange(0, len(mbox_source_layers)):
    layer = mbox_source_layers[i]
    num_bbox = num_bboxes[i]
    conf_layer = '{}_mbox_conf'.format(layer)
    voc_net.params[conf_layer][0].data.flat = sample_param(coco_net.params[conf_layer][0].data, len(coco_labelmap.item), len(voc_labelmap.item), num_bbox, maps)
    voc_net.params[conf_layer][1].data.flat = sample_param(coco_net.params[conf_layer][1].data, len(coco_labelmap.item), len(voc_labelmap.item), num_bbox, maps)
    loc_layer = '{}_mbox_loc'.format(layer)
    voc_net.params[loc_layer][0].data.flat = coco_net.params[loc_layer][0].data.flat
    voc_net.params[loc_layer][1].data.flat = coco_net.params[loc_layer][1].data.flat

layer = 'seg'
voc_net.params[layer][0].data.flat = sample_param(coco_net.params[layer][0].data, len(coco_labelmap.item), len(voc_labelmap.item), 1, maps)
voc_net.params[layer][1].data.flat = sample_param(coco_net.params[layer][1].data, len(coco_labelmap.item), len(voc_labelmap.item), 1, maps)
voc_net.save('/workspace/models/DES300x300_coco_to_voc.caffemodel')
