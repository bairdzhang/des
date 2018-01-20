#ifndef CAFFE_SEGGT_LAYER_HPP_
#define CAFFE_SEGGT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype> class SegGtLayer : public Layer<Dtype> {
public:
explicit SegGtLayer(const LayerParameter &param) : Layer<Dtype>(param) {
}
virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                        const vector<Blob<Dtype> *> &top);
virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                     const vector<Blob<Dtype> *> &top);

virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);
// virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
//                          const vector<Blob<Dtype> *> &top);
virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                          const vector<bool> &propagate_down,
                          const vector<Blob<Dtype> *> &bottom);
// virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
//                           const vector<bool> &propagate_down,
//                           const vector<Blob<Dtype> *> &bottom);

private:
Blob<Dtype> bias_multiplier_;
int outer_dim_, bias_dim_, inner_dim_, dim_;

protected:
SegGtParameter seggt_param_;
int background_label_id_;
bool use_difficult_gt_;
int num_gt_;
int num_;
int height_;
int width_;
bool non_exclusive_;
int num_class_;
Dtype *min_size = NULL;
};

} // namespace caffe

#endif // CAFFE_BIAS_LAYER_HPP_
