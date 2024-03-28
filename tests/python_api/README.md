## Set environmental variables
`export ROCAL_DATA_PATH=/Absolute/Path/Of/MIVisionX-data/``

## Reader Pipeline tests:
* To test all the reader pipelines with a single script
* The default value of number of gpu's is "1" & display is "ON" by default
./readers_test_file.sh

### Options:
* To test a single reader / multiple reader pipelines: Use the same script `readers_test_file.sh` as above and make the respective " Pipeline " to test equal to "1"

```
unit_test = 1
coco_pipeline = 0
```

* Example : To run COCO Pipeline
```
unit_test=0
coco_pipeline=1
caffe_reader=0
caffe2_reader=0
tf_classification_reader=0
tf_detection_pipeline=0
```

* To set number of GPUs/display/backend:
```
./readers_test_file.sh -n <number_of_gpus> -d <true/false>
./readers_test_file.sh -n "1" -d "false" -b "cpu"
```

## Augmentation + Reader Tests:
* This runs all augmentations and readers and compare them with golden outputs for both backends (HIP/HOST)

`./unit_tests.sh`

## Command line for individual files: 
* Test a single reader pipeline
* Example: COCO Pipeline
```
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

    # coco_pipeline.py
    # By default : cpu backend, NCHW format , fp32
    # Annotation must be a json file

    # For printing all the arguments that can be passed from user
    python$ver coco_pipeline.py -h

    python$ver coco_pipeline.py --image-dataset-path $data_dir --json-path $json_path --batch-size $batch_size --display --rocal-gpu --NHWC \
        --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 1 2>&1 | tee -a run.log.coco_pipeline.txt
```
### [ NOTE: Refer parse_config.py for more info on other args]
