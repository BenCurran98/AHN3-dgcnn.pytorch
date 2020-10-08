# AHN3-dgcnn.pytorch
Semantic segmentation of AHN3 point clouds using a [PyTorch implementation](https://github.com/AnTao97/dgcnn.pytorch) of Dynamic Graph CNN for Learning on Point Clouds (DGCNN)

Compared to the original PyTorch implementation, several changes are made in this repository to enable training and testing DGCNN on [AHN3 point cloud data](https://downloads.pdok.nl/ahn3-downloadpage/):

- prepare_data/extract_ahn3_annotations.py: convert AHN3 point clouds into S3DIS data format (.npy). The AHN3 dataset is already converted from .las to .txt in CloudCompare.
- data.py: add *S3DISDataset* class to load points block by block, instead of using large hdf5 files.
- main_semseg.py: add code to output segmented point cloud with labels (.txt), which can be used for later visualization.
- postprocess_data: add code for the multi-scale combination, by using results generated with different block sizes and *k* values in DGCNN.

Detailed description and results can be found in the [project report](http://resolver.tudelft.nl/uuid:492d2981-35ea-4cff-bc5a-eb75d06fc2dc).