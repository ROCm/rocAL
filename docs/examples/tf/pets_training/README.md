## Running Pets Training Example

### Building the required TF Rocm docker
* Use the instructions in the [docker section](https://github.com/ROCm/rocAL/docker) to build the required [Tensorflow docker](https://github.com/ROCm/rocAL/docker/rocal-with-tensorflow.dockerfile)
* Upgrade pip to the latest version.

### Running the training

* For first run, to setup dataset, edit "train.py" and set "DATASET_DOWNLOAD_AND_PREPROCESS = True"
* For subsequent runs, after the dataset has already been downloaded and preprocessed, set "DATASET_DOWNLOAD_AND_PREPROCESS = False"

To run this example for the first run or subsequent runs, just execute:
```
python3 train.py
```
