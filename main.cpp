#include<iostream>
#include<fstream>

#include "blob.hpp"

bool test_case(path, idx){
    Blob input_blob(path_input);
    Blob res_blob(path_res);
    Blob output_blob;
    auto t_start = hclock::now();

    input_blob.to_gpu();
    LRNLayer lru_layer;
    lru_layer.to_gpu();
    lru_layer.CrossChannelForward_gpu(input_blob, res_blob);
    res_blob.to_cpu();

    auto t_end = hclock::now();
    overall_time += std::chrono::duration_cast< milliseconds>(t_end - t_start); 
    bool is_pass = (output_blob == res_blob);

    return is_pass;
}

int main(){
    int case_num = 1;
    for(int i=0;i<case_num;i++){
        bool is_pass = test_case(path, i);
    }
}
