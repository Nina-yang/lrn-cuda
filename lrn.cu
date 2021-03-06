#include <cuda.h>

#include "lru_layer.hpp"
#include "utils.hpp"


// TODO: check if it would be faster to just put it into the previous kernel.
__global__ void LRNComputeOutput(const int nthreads, const float* const in,
    const float* const scale, const float negative_beta, float* const out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}


__global__ void LRNFillScale(const int nthreads, const float* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const float alpha_over_size,
    const float k, float* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const float* const in_off = in + offset;
    float* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    float accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}


void LRNLayer::CrossChannelForward_gpu(const Blob* bottom, Blob* top){
  // First, compute scale
  const float* bottom_data = bottom->gpu_ptr;
  float* top_data = top->gpu_ptr;
  float* scale_data = scale_->gpu_ptr;

  int n_threads = num_ * height_ * width_;

  LRNFillScale<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_, channels_, height_, width_, size_,
      alpha_ / size_, k_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom->count();

  LRNComputeOutput<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
}


void LRNLayer::to_gou(){
  this->scale_.to_gpu();
}
