#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author      : changfeng3168
# @time        : 22/11/05 17:22:48
# @description : find the atlas image with minest mse with mean image
# @reference   : 

import numpy as np
import os
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *

dir_path ='./nnunet/data/nnUNet_preprocessed/Task160_CandiBrainSegmentation/nnUNetData_plans_v2.1_stage0'

avg = np.zeros((128,160,160))

for file in tqdm(subfiles(dir_path, suffix = '.npy')):
    data = np.load(file)[0]
    avg = avg + data

avg = avg*1.0 / len(subfiles(dir_path, suffix = '.npy'))


def mse_error(x,y):
    return ((x-y)**2).mean()

min_mse_error = 1e+9
file_name = ''

for file in tqdm(subfiles(dir_path, suffix = '.npy')):
    data = np.load(file)[0]

    _mse_error = mse_error(data, avg)

    if _mse_error < min_mse_error: 
        min_mse_error = _mse_error
        file_name = file

print(file_name, min_mse_error) 

# take 025 as atlas