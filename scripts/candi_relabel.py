#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author      : changfeng3168
# @time        : 22/10/20 17:01:18
# @description : only use some classes of candi data
# @reference   : rename public_label = [2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60] to range(1,29)

import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *

dir_path = './data/candi/raw_data/data_cropped'
imagesTr = join(dir_path, 'imagesTr')
imagesTs = join(dir_path, 'imagesTs')
labelsTr = join(dir_path, 'labelsTr')
labelsTs = join(dir_path, 'labelsTs')

target_imagesTr = './data/candi/raw_data/preprocessed_label/imagesTr'
target_imagesTs = './data/candi/raw_data/preprocessed_label/imagesTs'
target_labelsTr = './data/candi/raw_data/preprocessed_label/labelsTr'
target_labelsTs = './data/candi/raw_data/preprocessed_label/labelsTs'

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)
maybe_mkdir_p(target_labelsTs)


good_labels = np.array([2,3,4,7,8,10,11,12,13,14,15,16,17,18,28,41,42,43,46,47,49,50,51,52,53,54,60])

print("preprocess for train set...")
for file in tqdm(os.listdir(labelsTr)):
    case_id = file.split('.')[0]

    label_path = os.path.join(labelsTr,file)

    ds = sitk.ReadImage(label_path)

    label_array = sitk.GetArrayFromImage(ds)

    result = np.zeros_like(label_array)
    for i, label in enumerate(good_labels):
        result[label_array==label] = (i+1)

    assert (np.unique(result) == np.array(range(28))).all()
    
    out = sitk.GetImageFromArray(result)
    out.SetSpacing(ds.GetSpacing())
    out.SetOrigin(ds.GetOrigin())
    out.SetDirection(ds.GetDirection())

    sitk.WriteImage(out, os.path.join(target_labelsTr,file) )
    os.system('cp {}/{} {}'.format(imagesTr, file, target_imagesTr))

print("preprocess for test set...")
for file in tqdm(os.listdir(labelsTs)):
    case_id = file.split('.')[0]

    label_path = os.path.join(labelsTs,file)

    ds = sitk.ReadImage(label_path)

    label_array = sitk.GetArrayFromImage(ds)

    result = np.zeros_like(label_array)
    for i, label in enumerate(good_labels):
        result[label_array==label] = (i+1)

    assert (np.unique(result) == np.array(range(28))).all()
    
    out = sitk.GetImageFromArray(result)
    out.SetSpacing(ds.GetSpacing())
    out.SetOrigin(ds.GetOrigin())
    out.SetDirection(ds.GetDirection())

    sitk.WriteImage(out, os.path.join(target_labelsTs,file) )
    os.system('cp {}/{} {}'.format(imagesTs, file, target_imagesTs))