## Set environmental variables

``export ROCAL_DATA_PATH=/Absolute/Path/Of/MIVisionX-data/``

### NOTE: Refer parse_config.py for more info on other args. This script is used with the `readers_test_file.sh` and `unit_tests.sh`

## Reader Pipeline Tests

* To test all the reader pipelines with a single script. The `readers_test_file.sh` tests the following cases:
  * unit test
  * coco reader
  * caffe reader
  * caffe2 reader
  * tf classification reader
  * tf detection reader
* The default value of number of gpu's is "1" & display is "ON" by default
`./readers_test_file.sh`

### Options

* To test a single reader / multiple reader pipelines: Use the same script `readers_test_file.sh` as above and make the respective " Pipeline " to test equal to "1"

```shell
unit_test = 1
coco_reader = 0
```

* Example : To run COCO Pipeline

```shell
unit_test=0
coco_reader=1
caffe_reader=0
caffe2_reader=0
tf_classification_reader=0
tf_detection_reader=0
```

* To set options:
  * Number of GPUs `-n`
  * display `-d`
  * backend `-b`

```shell
./readers_test_file.sh -n <number_of_gpus> -d <true/false>
./readers_test_file.sh -n "1" -d "false" -b "cpu"
```

## Augmentation + Reader Tests

* This runs all augmentations and readers and compare them with golden outputs for both backends (HIP/HOST)

`./unit_tests.sh`

* This test also runs the `image_comaprison.py` script at the end to compare the results with the golden outputs from [MIVisionX-data](https://www.github.com/ROCm/MIVisionX-data).

### Command line for individual files

* Test a single reader pipeline
* Example: COCO Pipeline

```shell
    # Mention the number of gpus
    gpus_per_node=4

    # Mention Batch Size
    batch_size=10

    # python version
    ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\.\2/')

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/coco/coco_10_img/val_10images_2017/

    # Mention json path
    json_path=$ROCAL_DATA_PATH/coco/coco_10_img/annotations/instances_val2017.json

    # coco_reader.py
    # By default : cpu backend, NCHW format , fp32
    # Annotation must be a json file

    # For printing all the arguments that can be passed from user
    python$ver coco_reader.py -h

    python$ver coco_reader.py --image-dataset-path $data_dir --json-path $json_path --batch-size $batch_size --display --rocal-gpu --NHWC \
        --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 1 2>&1 | tee -a run.log.coco_reader.txt
```

## Decoder Test

This test runs a simple decoder pipeline making use of a image file reader and resizing the given input images to 300x300.

It uses the [AMD-TinyDataSet](../../data/images/AMD-tinyDataSet/) by default, unless otherwise specified by user.

### Options

```bash
backend: <cpu/gpu>
Input image folder: <Path to image folder>
```

* Usage:

```shell
python3 decoder.py
python3 decoder.py gpu <path to image folder>
```

## External source reader test

This test runs a pipeline making use of the external source reader in 3 different modes. It uses coco2017 images by default.

* Mode 0 - Filename
* Mode 1 - Raw compressed images
* Mode 2 - JPEG reader with OpenCV

* Usage:

```bash
python3 external_source_reader.py
```

## Audio Unit Test

To run the Audio unit test with all test cases. Follow the steps below

```bash
export ROCAL_DATA_PATH=<Absolute_path_to_MIVisionX-data>
```
To run the audio unit test and verify the correctness of the outputs

```bash
python3 audio_unit_test.py
```
To pass the audio data path, batch size, and run a particular test case use the following command

```bash
python3 audio_unit_test.py --audio_path=<path_to_data> --test_case <case(0-1)> --batch-size <batch_size>
```

**Available Test Cases**
* Case 0 - Audio Decoder
* Case 1 - PreEmphasis Filter
