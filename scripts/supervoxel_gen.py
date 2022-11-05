import edt
import os
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import json
import shutil
from skimage import segmentation
from tqdm import tqdm

target_dir = './nnunet/data/nnUNet_preprocessed/Task160_CandiBrainSegmentation/supervoxels'
data_dir = './nnunet/data/nnUNet_preprocessed/Task160_CandiBrainSegmentation/nnUNetData_plans_v2.1_stage0/'

maybe_mkdir_p(target_dir)

nregions = 2000
compactness = 0.3

for file in tqdm(subfiles(data_dir, join = False, suffix = '.npy')):
    file_path = join(data_dir, file)

    image = np.load(file_path)
    mask = image[1]
    image = image[0]
    tmp_image = image.copy()

    image[image<image.mean()] = image.mean()

    image[image>(0.75*image.mean()+0.25*image.max())] = 0.75*image.mean()+0.25*image.max()

    image = ((image-image.min())/(image.max()-image.min())*255).astype(np.uint8)

    label = segmentation.slic(image, n_segments = nregions, compactness = compactness, multichannel = False, start_label=1)

    label = np.transpose(label, (1,2,0))
    dt = edt.edt(
    label, anisotropy=(0.9375, 0.9375,1.5), 
    #label, anisotropy=(1, 1, 1), 
    black_border=True, order='C',  # F? for LPBA40
    parallel=8 # number of threads, <= 0 sets to num cpu
    ) 
    label = np.transpose(label, (2,0,1))
    dt = np.transpose(dt, (2,0,1))


    #dt[tmp_image<4e-2]=0
    dt = dt- dt.min()

    tmp_dt = sitk.GetImageFromArray(dt)
    #sitk.WriteImage(tmp_dt, join(target_dir, 'tmp_dt.nii.gz'))

    sitk.WriteImage(tmp_dt, join(target_dir, file.replace('.npy','.nii.gz')))
    np.save(join(target_dir, file), dt)