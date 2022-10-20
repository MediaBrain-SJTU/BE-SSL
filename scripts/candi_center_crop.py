import SimpleITK as sitk
import numpy as np
import os
import random
from tqdm import tqdm

dir_path = './data/candi/raw_data/data/'
target_dir_path = './data/candi/raw_data/data_cropped/'
os.system('mkdir {}'.format(target_dir_path))

HC_images = [dir_path+'HC_'+str(i).zfill(3)+'/HC_'+str(i).zfill(3)+'_procimg.nii.gz' for i in range(1,30)]
HC_labels = [dir_path+'HC_'+str(i).zfill(3)+'/HC_'+str(i).zfill(3)+'_seg.nii.gz' for i in range(1,30)]
BPDwithoutPsy_images = [dir_path+'BPDwoPsy_'+str(i).zfill(3)+'/BPDwoPsy_'+str(i).zfill(3)+'_procimg.nii.gz' for i in range(30,65)]
BPDwithoutPsy_labels = [dir_path+'BPDwoPsy_'+str(i).zfill(3)+'/BPDwoPsy_'+str(i).zfill(3)+'_seg.nii.gz' for i in range(30,65)]
BPDwithPsy_images = [dir_path+'BPDwPsy_'+str(i).zfill(3)+'/BPDwPsy_'+str(i).zfill(3)+'_procimg.nii.gz' for i in range(65,84)]
BPDwithPsy_labels = [dir_path+'BPDwPsy_'+str(i).zfill(3)+'/BPDwPsy_'+str(i).zfill(3)+'_seg.nii.gz' for i in range(65,84)]
SS_images = [dir_path+'SS_'+str(i).zfill(3)+'/SS_'+str(i).zfill(3)+'_procimg.nii.gz' for i in range(84,104)]
SS_labels = [dir_path+'SS_'+str(i).zfill(3)+'/SS_'+str(i).zfill(3)+'.seg.nii.gz' for i in range(84,104)]

images = HC_images + BPDwithoutPsy_images + BPDwithPsy_images + SS_images
labels = HC_labels + BPDwithoutPsy_labels + BPDwithPsy_labels + SS_labels
image_label_dict = [{'image':images[i],'label':labels[i]} for i in range(103)]

random.shuffle(image_label_dict)

imagesTr = os.path.join(target_dir_path, 'imagesTr')
imagesTs = os.path.join(target_dir_path, 'imagesTs')
labelsTr = os.path.join(target_dir_path, 'labelsTr')
labelsTs = os.path.join(target_dir_path, 'labelsTs')
os.system('mkdir -p {}'.format(imagesTr))
os.system('mkdir -p {}'.format(imagesTs))
os.system('mkdir -p {}'.format(labelsTr))
os.system('mkdir -p {}'.format(labelsTs))

for item in tqdm(image_label_dict[:int(len(image_label_dict)*0.8)]):
    image_path, label_path = item['image'], item['label']

    case_id = image_path.split('/')[-1].split('_')[1]
    
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    size = image.GetSize()

    cropped_image = sitk.Extract(image, (160,160,128),(48,48,size[2]//2-64))
    cropped_label = sitk.Extract(label, (160,160,128),(48,48,size[2]//2-64))

    sitk.WriteImage(cropped_image, os.path.join(imagesTr, case_id+'.nii.gz') )
    sitk.WriteImage(cropped_label, os.path.join(labelsTr, case_id+'.nii.gz') )

for item in tqdm(image_label_dict[int(len(image_label_dict)*0.8):]):
    image_path, label_path = item['image'], item['label']

    case_id = image_path.split('/')[-1].split('_')[1]
    
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    size = image.GetSize()

    cropped_image = sitk.Extract(image, (160,160,128),(48,48,size[2]//2-64))
    cropped_label = sitk.Extract(label, (160,160,128),(48,48,size[2]//2-64))

    sitk.WriteImage(cropped_image, os.path.join(imagesTs, case_id+'.nii.gz') )
    sitk.WriteImage(cropped_label, os.path.join(labelsTs, case_id+'.nii.gz') )