* This example shows how to run training using pytorch and ToyNet with 2 classes
* Use a dataset with 2 classes

### Building the required Pytorch Rocm docker
* Use the instructions in the [docker section](https://github.com/ROCm/rocAL/docker) to build the required [Pytorch docker](https://github.com/ROCm/rocAL/docker/rocal-with-pytorch.dockerfile)
* Upgrade pip to the latest version.
* Run requirements.sh to install the required packages.

### To run the sample:
* Install rocal_pybind

```
python3 train.py <image_folder> <cpu/gpu> <batch_size>
```
* rocal device can be cpu/gpu.
