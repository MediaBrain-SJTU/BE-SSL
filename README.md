# BE-SSL

![1](https://user-images.githubusercontent.com/24731236/179522219-2913d410-a50d-4d56-8ce0-dfa1eda17b93.png)


**Abstract**: To alleviate the demand for a large amount of annotated data by deep learning methods, this paper explores self-supervised learning (SSL) for brain structure segmentation. Most SSL methods treat all pixels equally, failing to emphasize the boundaries that are important clues for segmentation. We propose Boundary-Enhanced Self-Supervised Learning (BE-SSL), leveraging supervoxel segmentation and registration as two related proxy tasks. The former task enables capture boundary information by reconstructing distance transform map transformed from supervoxels. The latter task further enhances the boundary with semantics by aligning tissues and organs in registration. Experiments on CANDI and LPBA40 datasets have demonstrated that our method outperforms current SOTA methods by  0.89% and 0.47%, respectively.

Code will be made available soon.

## Get Started

### Environment

> cd nnunet \
> pip install -e . \
> pip install edt


### Files Preparation
Download CANDI dataset (Img+Seg+Reg_V1.2) from [candi_share](https://www.nitrc.org/projects/candi_share). The target dataset should be formulated as:

```
|---nnunet 
|---data 
|   |---candi 
|       |---raw_data
|           |---SchizBull_2008_BPDwoPsy_segimgreg_V1.2.tar
|           |---SchizBull_2008_BPDwPsy_segimgreg_V1.2.tar
|           |---SchizBull_2008_HC_segimgreg_V1.2.tar
|           |---SchizBull_2008_SS_segimgreg_V1.2.tar
|---README.md
```

### Quick Start

#### file preparation
> python scripts/candi_unpack.py \
> python scripts/candi_cropped.py \
> python scripts/candi_relabel.py

raw data format:
> python nnunet/nnunet/dataset_conversion/Task160_Candi.py

plan and data preprocess:
> nnUNet_plan_and_preprocess -t 160

unpack npz data:
> python nnunet/nnunet/training/dataloading/dataset_loading.py

find atlas:
> python scripts/find_atlas.py

supervoxel generation:
> python scripts/supervoxel_gen.py

### Training
BE-SSL preTrain:
> nnUNet_train 3d_fullres Candi_preTrain {task-id} {fold}
BE-SSL finetune:
> nnUnet_train 3d_fullres Candi_finetune {task-id} {fold}

### Results
<img src=https://user-images.githubusercontent.com/24731236/200122862-3501aedc-03f8-4dee-816e-461632fe6859.png width="600px"/>


### Visualization
<img src=https://user-images.githubusercontent.com/24731236/200122853-3b4bf124-e0a8-41c5-bb5e-46c65fd41f53.png width="800px"/>


### Acknowledgement
We conduct experiments on the basis of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).


