# BE-SSL

![1](https://user-images.githubusercontent.com/24731236/179522219-2913d410-a50d-4d56-8ce0-dfa1eda17b93.png)


**Abstract**: To alleviate the demand for a large amount of annotated data by deep learning methods, this paper explores self-supervised learning (SSL) for brain structure segmentation. Most SSL methods treat all pixels equally, failing to emphasize the boundaries that are important clues for segmentation. We propose Boundary-Enhanced Self-Supervised Learning (BE-SSL), leveraging supervoxel segmentation and registration as two related proxy tasks. The former task enables capture boundary information by reconstructing distance transform map transformed from supervoxels. The latter task further enhances the boundary with semantics by aligning tissues and organs in registration. Experiments on CANDI and LPBA40 datasets have demonstrated that our method outperforms current SOTA methods by  0.89\% and 0.47\%, respectively.

Code will be made available soon.

## Get Started

### Environment

### Files Preparation
Download CANDI dataset (Img+Seg+Reg_V1.2) from [candi_share](https://www.nitrc.org/projects/candi_share). The target dataset should be formulated as:
> |---nnunet
> |---data
> |   |---candi
> |       |---raw_data
> |---README.md


### Quick Start

### Training

### Results

### Visualization

### Acknowledgement
We borrow codes from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

