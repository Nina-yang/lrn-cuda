import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import os 
import shutil
import os
import argparse



def generate_one_case():
    a_b = 1
    a_c = 128
    a_h = 256
    a_w = 256
         
    a = torch.rand((a_b, a_c, a_h, a_w)).float()
    
    return a


lrn = nn.LocalResponseNorm(size=3,alpha=1,beta=1,k=2)

def generate_cases(test_case_dir, case_nums, tensor_size):
    tensor_size = 
    if os.path.exists(test_case_dir):
        shutil.rmtree(test_case_dir)
    os.mkdir(test_case_dir)
    for i in tqdm(range(case_nums)):
        data_type = random.sample(data_type_list, k =1)[0]
        a = generate_one_case()
        b = lrn(a) 
        
        # print("a: ", a.shape)
        # print("b: ", b.shape)

        # append meta info, i.e, shape to the start of the tensor
        a = a.flatten()
        b = b.flatten()

        a.numpy().astype(data_type).tofile('./{}/case_{}_{}_a.bin'.format(test_case_dir,i,data_type))
        b.numpy().astype(data_type).tofile('./{}/case_{}_{}_b.bin'.format(test_case_dir,i,data_type))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--out_directory', type = str, default='./test_case', help='Directory to save test cases')
    parser.add_argument('-n','--case_nums', type = int, default = 10, help='Number of test cases')
    parser.add_argument('-r','--random_seed', type = int, default= 42, help = 'Random seed')
    args = parser.parse_args()
    
        
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    generate_cases(args.out_directory, args.case_nums)
