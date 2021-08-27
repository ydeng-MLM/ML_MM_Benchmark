# Benchmarking ADM problems with various DL structures
This repository stores implemention of paper [Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials]() 

It includes a suit of ADM data set benchmarks along with implementation of various ready-to-use deep learning architectures (MLP, Transformer, MLP-Mixer) for scientific computation problem and a handful of utility functions.

## Data Sets
[TO ADD DESCRIPTION]

## Requirements
| Package | Version |
|:---------------------------------------------:|:------------------------------------------------------------------:|
| Python | \>=3.7 |
| Pytorch | \>= 1.3.1 |
| Numpy  | \>=1.17.4 |
| Pandas | \>=0.25.3 |
| Tensorboard | \>=2.0.0 |
| Tqdm| \>=4.42.0 |
| Sklearn | \>=0.22.1|
| Matplotlib | \>= 3.1.3|
### Environment
1. The detailed conda environment is packaged in [.yml file](./demo/environment_droplet.yml).
2. Add the [Benchmarking folder](./Benchmarking%20Algorithms) as one of the source directory to make utils and Simulated_Dataset folders 
visible runtime

## Features 
* Access to various ADM data sets 
* **Off-the-shelf implementation** of MLP, Transformer and MLP-Mixer with high individuality
* Utilities for **data preprocessing and preparation** for downstream deep learning tasks
* Utilities for **plotting** and easy analysis of results



## Usage

### Access to Data Sets
1. ADM Data Set. Please download and unzip from the [website](https://www.atom3d.ai/).
2. Particle Data Set. Please download and unzip from the [website](https://www.atom3d.ai/).
3. Color Data Set. Please download and unzip from the [website](https://www.atom3d.ai/).

### Loading and Spliting the Data Sets
```
import atom3d.datasets as da
dataset = da.load_dataset(PATH_TO_DATASET, {'lmdb','pdb','silent','sdf','xyz','xyz-gdb'})
print(len(dataset))  # Print length
print(dataset[0].keys())  # Print keys
```


## Support

Please file an issue [here **(CHANGE LINK BEFORE SUBMISSION)**](https://github.com/drorlab/atom3d/issues).

## License

The project is licensed under the [MIT license](https://github.com/drorlab/atom3d/blob/master/LICENSE).

Please cite this work if some of the code or datasets are helpful in your scientific endeavours. For specific datasets, please also cite the respective original source(s), given in the preprint.
