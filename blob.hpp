#ifndef BLOB_H_
#define BLOB_H_


#include <stdio.h>
#include <time.h>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

#include "utils.cu"

class Blob{
    public:
        Blob(const char* path){
            cpu_ptr = read_data_from_file(path);
        }

        Blob(){
            cpu_ptr = (void *)malloc(size_ * sizeof(float));
        }

        ~Blob(){
            if(cpu_ptr){
                free(cpu_ptr);
            }
        }

        bool operator == (const Blob& t) const{
            size_t size = this->size_;
            for (size_t i = 0; i < size; ++i) {
                if (*((float*)this->cpu_ptr + i) != *((float*)t->cpu_ptr + i)) 
                    return false;
            }  
            return true;
        }

        float* read_data_from_file(const char* path){
            std::ifstream input;
            input.open(path, std::ios::in | std::ios::binary);
            float * arr = (float *)malloc(size_ * sizeof(float));
            input.read((char*)arr, size * sizeof(float));
            input.close();
            return arr;
        }

        int count(){
            return this->size_;
        }

        int B = 1;
        int C = 128;
        int H = 256;
        int W = 256;
        size_t size_ = B * C * H * W;

        void* cpu_ptr;
        void* gpu_ptr;
        void to_cpu(){
            if(cpu_ptr){
                gpu_memcpy(size_, gpu_ptr, cpu_ptr);
            }
        };
        void to_gpu(){
            if(gpu_ptr == NULL){
                CUDA_CHECK(cudaMalloc(&gpu_ptr, size_));
                gpu_memset(size_, 0, gpu_ptr);
            }
            gpu_memcpy(size_, cpu_ptr, gpu_ptr);
        };

};

#endif
