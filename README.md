# Benchmarking ADM problems with various DL structures
This is the repository for NeuIPS paper [...] 

3 physical problems + multiple ML architectures benchmarking

## Features (We have to make up some cool features)
* Access to several datasets involving 3D molecular structure. 
* LMDB data format for storing lots of molecules (and associated metadata).
* Utilities for splitting/filtering data based on many criteria.

## Data Sets

## Usage
### Downloading a dataset

From python:
```
import atom3d.datasets as da
da.download_dataset('lba', PATH_TO_DATASET) # Download LBA dataset.
```

Or, download and unzip from the [website](https://www.atom3d.ai/).

### Loading a dataset

From python:
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
