#include <algorithm>
#include <vector>

#include "caffe/layers/seggt.hpp"

namespace caffe {

template <typename Dtype>
void SegGtLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
        seggt_param_ = this->layer_param_.seggt_param();
        background_label_id_ = seggt_param_.background_label_id();
        use_difficult_gt_ = seggt_param_.use_difficult_gt();
        non_exclusive_ = seggt_param_.non_exclusive();
        num_class_ = non_exclusive_ ? seggt_param_.num_class() : 1;
        num_ = bottom[1]->shape(0);
        height_ = bottom[1]->shape(2);
        width_ = bottom[1]->shape(3);
        num_gt_ = bottom[0]->height();
        top[0]->Reshape(num_, num_class_, height_, width_);
        min_size = non_exclusive_ ? NULL :
                   (Dtype *)malloc(num_ * height_ * width_ * sizeof(Dtype));
}

template <typename Dtype>
void SegGtLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {
        num_ = bottom[1]->shape(0);
        height_ = bottom[1]->shape(2);
        width_ = bottom[1]->shape(3);
        num_gt_ = bottom[0]->height();
        top[0]->Reshape(num_, num_class_, height_, width_);
        if (min_size != NULL)
        {
                free(min_size);
        }
        min_size = non_exclusive_ ? NULL :
                   (Dtype *)malloc(num_ * height_ * width_ * sizeof(Dtype));
}

template <typename Dtype>
void SegGtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
        const Dtype *gt_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        memset(top_data, 0, top[0]->count() * sizeof(Dtype));
        map<int, vector<NormalizedBBox> > all_gt_bboxes;
        map<int, vector<NormalizedBBox> >::iterator all_gt_bboxes_i;
        GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                       &all_gt_bboxes);
        if (non_exclusive_ == false)
        {
                for (int i = 0; i < num_ * height_ * width_; ++i) {
                        min_size[i] = 10.0; // All size should <= 1.0, so 10.0 here means INF
                }
        }
        for (int i = 0; i < num_; ++i) {
                int shift1 = i * num_class_ * height_ * width_;
                if (non_exclusive_)
                {
                        for (int y_i = 0; y_i < height_; ++y_i)
                        {
                                for (int x_i = 0; x_i < width_; ++x_i)
                                {
                                        top_data[shift1 + y_i * width_ + x_i] = 1;
                                }
                        }
                }
                all_gt_bboxes_i = all_gt_bboxes.find(i);
                if (all_gt_bboxes_i == all_gt_bboxes.end()) {
                        continue;
                }
                const vector<NormalizedBBox> gt_bboxes = all_gt_bboxes_i->second;
                for (int j = 0; j < gt_bboxes.size(); ++j) {
                        const NormalizedBBox &gt_bbox = gt_bboxes[j];
                        Dtype xmin, ymin, xmax, ymax, label;
                        int xmin_idx, ymin_idx, xmax_idx, ymax_idx;
                        xmin = gt_bbox.xmin();
                        ymin = gt_bbox.ymin();
                        xmax = gt_bbox.xmax();
                        ymax = gt_bbox.ymax();
                        label = gt_bbox.label();
                        xmin_idx = width_ * xmin;
                        ymin_idx = height_ * ymin;
                        xmax_idx = width_ * xmax;
                        ymax_idx = height_ * ymax;
                        CHECK(xmin_idx >= 0);
                        CHECK(ymin_idx >= 0);
                        CHECK(xmax_idx <= width_);
                        CHECK(ymax_idx <= height_);
                        Dtype size = (ymax - ymin) * (xmax - xmin);
                        int channel_offset = non_exclusive_ ? label * width_ * height_ : 0;
                        for (int y_i = ymin_idx; y_i < ymax_idx; ++y_i) {
                                int shift2 = y_i * width_;
                                for (int x_i = xmin_idx; x_i < xmax_idx; ++x_i) {
                                        int idx_ = channel_offset + shift1 + shift2 + x_i;
                                        if (non_exclusive_ == true or size < min_size[idx_]) {
                                                if (non_exclusive_ == false)
                                                {
                                                        min_size[idx_] = size;
                                                }
                                                else
                                                {
                                                        top_data[shift1 + y_i * width_ + x_i] = 0;
                                                }
                                                top_data[idx_] = non_exclusive_ ? 1 : label;
                                        }
                                }
                        }
                }

        }
}

template <typename Dtype>
void SegGtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                     const vector<bool> &propagate_down,
                                     const vector<Blob<Dtype> *> &bottom) {
        NOT_IMPLEMENTED; // Do Nothing Here
}

#ifdef CPU_ONLY
STUB_GPU(SegGtLayer);
#endif

INSTANTIATE_CLASS(SegGtLayer);
REGISTER_LAYER_CLASS(SegGt);

} // namespace caffe
