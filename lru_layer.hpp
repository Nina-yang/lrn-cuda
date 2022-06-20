#ifndef LRN_LAYER_HPP_
#define LRN_LAYER_HPP_

#include <vector>
#include "blob.hpp"

class LRNLayer{
 public:
//   void CrossChannelForward_cpu(const Blob* bottom, Blob* top);
  void CrossChannelForward_gpu(const Blob* bottom, Blob* top);


  int size_ = 3;
  float alpha_ = 1.0;
  float beta_ = 1.0;
  float k_ = 2.0;
  int num_ = 1;
  int channels_ = 128;
  int height_ = 256;
  int width_ = 256;

  Blob scale_;
};

#endif  // LRN_LAYER_HPP_
