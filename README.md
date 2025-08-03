# Graph Attention-Driven Bayesian Deep Unrolling for Dual-Peak Single-Photon Lidar Imaging
This repository contains the official implementation of Graph Attention-Driven Bayesian Deep Unrolling for Dual-Peak Single-Photon Lidar Imaging.

## Dependencies
To set up the environment, simply run:
```
conda env create -f environment.yml
```
This will create a Conda environment with all necessary dependencies.

## Testing

We provide a synthetic Art scene dataset (PPP=4.0, SBR=4.0) based on the Middlebury dataset, located in run/Art_4.0_4.0.

To run the test:
```
cd run
python test_middlebury.py
```

## Training

To train the model from scratch, you first need to generate synthetic training data.

```
cd gen_data
bash download_data.sh 1
Python create_data_mpi_2obj 4 64
```
Once the data is generated, go to the training directory and start training:
```
cd train
Python Train.py
```
