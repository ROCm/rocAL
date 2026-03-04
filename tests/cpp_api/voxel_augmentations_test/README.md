# rocAL voxel augmentations test

This application verifies the functionality of the voxel (3D/volumetric) augmentation APIs in rocAL. It builds a pipeline that reads numpy files in NCDHW layout and runs the following operations:

- RandomObjectBbox (start_end and anchor_shape formats)
- ROIRandomCrop
- SliceFixed
- Log
- Flip / FlipFixed (with depth flag)
- Brightness / BrightnessFixed (with conditional execution)
- GaussianNoise / GaussianNoiseFixed (with conditional execution)

## Pre-requisites

- Ubuntu Linux, version `22.04` or later
- rocAL library
- ROCm Performance Primitives (RPP)
- MIVisionX

## Build Instructions

````bash
mkdir build
cd build
cmake ../
make
````

## Prepare Dataset

The test reads `.npy` files from a `voxel_numpy` folder under the `rocal_data` directory.

```
voxel_numpy_path (.npy files) : ${ROCAL_DATA_PATH}/rocal_data/voxel_numpy/
```

Set the environment variable before running:

```bash
export ROCAL_DATA_PATH=<absolute_path_to_data_directory>
```

## Running the Application

```bash
./voxel_augmentations_test <gpu=1/cpu=0> <batch_size>
```

| Argument | Description | Default |
|----------|-------------|---------|
| `gpu=1/cpu=0` | Processing device: 0 for CPU, 1 for GPU | 0 (CPU) |
| `batch_size` | Number of samples per batch | 2 |
