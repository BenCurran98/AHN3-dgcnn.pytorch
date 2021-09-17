# dgcnn.pytorch
A package containing functionality for training and inference using the *Dynamic Graph Convolutional Neural Net* (DGCNN) architecture.
Based off the *AHN3-dgcnn.pytorch* package. 

<p align="center">
    <img src="figures/DGCNNArchitecture.png"/>
</p>
<p align="center">
    DGCNN Architecture
</p>
<p align = "center">
    Obtained from *Widyaningrum, E.; Bai, Q.; Fajari, M.K.; Lindenbergh, R.C. Airborne Laser Scanning Point Cloud Classification Using the DGCNNDeep Learning Method. Remote Sens.2021, 13, 859. https://doi.org/10.3390/rs13050859*
</p>

## Data

### Pre-Processing
Data for this model is extracted from point clouds in either *LAS* or *HDF5* formats, and the functionality for processing these sources is mostly in *./prepare_data*
To load in a set of pointclouds from a directory `pointcloud_dir`, process them, and then save the processed pointclouds to another directory `out_dir`, the `load_pointcloud_dir` function can be used. Pointclouds are split into blocks or tiles for easier compute which can be randomly sampled. For example, the following code snippet loads the pointclouds in `pointcloud_dir`, splits it into 5 100mx100m tiles and converts the existing labels each pointcloud to labels specified by the mapping `CLASS_MAP`. The quantity `min_num` specifies the minimum number of points that should be present for each class in order for the block to be added to the data set. 
This will return a set of `data_batches` and `label_batches` containing the converted data.
```python
pointcloud_dir = "some_dir"
out_dir = "some_other_dir"
block_size = 100
sample_num = 5
CLASS_MAP = {
    1 : 0,
    3 : 1,
    4 : 2,
    10 : 3
}

min_num = 10

data_batches, label_batches = load_pointcloud_dir(pointcloud_dir, outdir, block_size = block_size, sample_num = sample_num, classes = CLASS_MAP, min_num = min_num)
```

### File Systems
For the model to have access to a dataset, it is necessary to format the data directory appropriately. Data stored in a "Datasets" directory should be stored as below. Here *Pointcloud_dir* contains the raw pointclouds, *Area* contains the "annotated" pointcloud data separated into tiles and sorted further into text files containing the points corresponding to each class label in that tile. The content in this directory is produced by the `extract_annotations` function in `process_data.py`. The directory `data_NPY` contains the data put into *.npy* files, which are passed into data loaders and fed as arrays into the function during training/inference.

```bash
Datasets
|--Pointcloud_dir
|--Area
|----data_folder
|------raw pointcloud data in *.txt* format
|----processed_data_folder
|------Area_n
|--------Area
|----------Area_n.txt (contains all data for this block)
|----------Annotations
|------------Separate text files for points corresponding to each class in this block
|--data_NPY
|----All *.npy* data files used directly in training
```

The data loaders in this package use files in *.npy* format, so the `data_batches` and `data_labels` have to be converted into this form. 

## Model Interface
The *DGCNN* model is implemented using *PyTorch* by the `DGCNN` class in `model.py`. The architecture mostly follows the above figure (save for some differences in the weight dimensions. To define a model, specify the number of classes `num_classes`, the number of nearest neighbours `k` to compute in the *EdgeConv* layers, the "embedding dimensions' `emb_dims` which is the dimension the gloabl feature data is projected into, and `cuda` which indicates whether or not to use `CUDA` in model inference. 
```python
num_classes = 5
k = 30
emb_dims = 1024
cuda = False
dgcnn = DGCNN(num_classes, k, dropout = dropout, emb_dims = emb_dims, cuda = cuda)
```

To directly get a prediction from the model given an `Bx3xN` tensor of points (`B` is the batch size, `N` is the number of points), one can pass the matrix directly into the model class.

```python
N = 1000
B = 1
data = torch.rand(B, 3, N)
logits = model(data)
predictions = logits.max(dim = 2)[1]
```

Training can be performed using the `train` function in `train.py`, which has a selection of configurable hyperparameters, directory locations and logging options. 
As a simple example, one may specify the model parameters above, as well as the location of a dataset given as a set of *numpy* *.npy* files, specify the number of classes, the block size, number of points per block to sample, and the number of epochs to train for, as given below. Also requirted is an `IOStream` where all logging data is streamed to.
```python
num_classes = 5
data_dir = "path/to/npy_data/"
block_size = 30.0 # metres
num_points = 5000
epochs = 30
k = 30
log_file = "path/to/my/logfile.log"
io = IOStream(log_file)
train(k, io, data_dir = data_dir, num_classes = num_classes, block_size = block_size, num_points = num_points, epochs = epochs)
```

Inference on a data set can be performed using the `test` function in `test.py`. Similarly to `train`, this can be called by specifying a data location, model and sampling parameters, etc.

Predictions made by the model via the `test` function are saved in a predicitons folder in text format. These can be extracted and saved in *LAS* files using the `get_predictions` function, e.g.
```python
prediction_file = "path/to/predicitons.txt"
las_file = "path/to/predictions.las"
points, labels = get_predictions(prediction_file, las_file)
```