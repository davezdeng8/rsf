# RSF: Optimizing Rigid Scene Flow From 3D Point Clouds Without Labels

## Installation
Our code has the following dependencies:

-PyTorch

-PyTorch3D

-Mayavi

-Scipy

-Matplotlib

-Shapely

Create the conda environment:

```angular2html
conda env create -f environment.yml
conda activate segflow3
```

## Data
Download the data from the following links:

-[StereoKITTI](https://drive.google.com/file/d/1j4-0QINSmqJYseIONSK2hbW_-9DLQ8Ak/view?usp=sharing)

-[LidarKITTI](https://drive.google.com/file/d/1FmBD_c5q0O7JMd9ufKyHkuMg7V0bRR_f/view?usp=sharing)

-[nuScenes](https://drive.google.com/file/d/1mCjDqJzaMdW0iiM2N2J5BNvo04dAvTbx/view)

Extract them into rsf's parent directory 
(rsf and dataset directories are in the same directory)

## Usage
Run 'without_learning2.py' to produce results for StereoKITTI. 
Change the cfg argument to run the other datasets. 
