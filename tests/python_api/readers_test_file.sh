#!/bin/bash

if [[ $# -gt 0 ]]; then
    helpFunction()
    {
    echo ""
    echo "Usage: $0 [-n number_of_gpus] [-d dump_outputs<true/false>] [-b backend<cpu/gpu>] [-p print_tensor<true/false>]"
    echo -e "\t-n Description of what is the number of gpus to be used"
    echo -e "\t-d Description of what is the display param"
    echo -e "\t-d Description of what is the print tensor param"
    exit 1 # Exit script after printing help
    }

    while getopts "n:d:b:p:" opt
    do
        echo "In while loop"
        echo $opt
        case "$opt" in
            n ) number_of_gpus="$OPTARG" ;;
            d ) dump_outputs="$OPTARG" ;;
            b ) backend="$OPTARG" ;;
            p ) print_tensor="$OPTARG" ;;
            ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
        esac
    done

    # Print helpFunction in case parameters are empty

    if [ -z "$backend" ];
    then
        backend_arg=no-rocal-gpu
    else
        if [[ $backend == "cpu" || $backend == "CPU" ]]; then
            backend_arg=no-rocal-gpu
        elif [[ $backend == "gpu" || $backend == "GPU" ]]; then
            backend_arg=rocal-gpu
        fi
    fi

    if [ -z "$number_of_gpus" ];
    then
        gpus_per_node=1
    else
        gpus_per_node=$number_of_gpus
    fi


    if [ -z "$dump_outputs" ];
    then
        display_arg=display #True by default
    else
        if [[ $dump_outputs == "true" || $dump_outputs == "True" ]]; then
            display_arg=display
        elif [[ $dump_outputs == "false" || $dump_outputs == "False" ]]; then
            display_arg=no-display
        fi
    fi

    if [ -z "$print_tensor" ];
    then
        print_tensor_arg=print_tensor #True by default
    else
        if [[ $print_tensor == "true" || $print_tensor == "True" ]]; then
            print_tensor_arg=print_tensor
        elif [[ $print_tensor == "false" || $print_tensor == "False" ]]; then
            print_tensor_arg=no-print_tensor
        fi
    fi


    echo $display_arg
    echo $backend_arg
    echo $print_tensor_arg


else
    #DEFAULT ARGS
    gpus_per_node=1
    display_arg=display
    backend_arg=no-rocal-gpu #CPU by default
    print_tensor_arg=no-print_tensor
fi

echo $ROCAL_DATA_PATH
CURRENTDATE=`date +"%Y-%m-%d-%T"`

# Mention Batch Size
batch_size=10

# python version
ver=$(python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";)


####################################################################################################################################
# USER TO MAKE CHANGES HERE FOR TEST
# Make the respective " Pipeline " to test equal to 1
unit_test=0
coco_reader=0
caffe_reader=0
caffe2_reader=0
tf_classification_reader=0
tf_detection_reader=0
video_pipeline=0
web_dataset_reader=1
####################################################################################################################################




####################################################################################################################################
if [[ unit_test -eq 1 ]]; then

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/rocal_data/images_jpg/labels_folder/


    # unit_test.py
    # By default : cpu backend, NCHW format , fp32
    # Please pass image_folder augmentation_nanme in addition to other common args
    # Refer rocAL_api_python_unitest.py for all augmentation names
    python"$ver" unit_test.py \
        --image-dataset-path $data_dir \
        --augmentation-name snow \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 2 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ coco_reader -eq 1 ]]; then

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/rocal_data/coco/coco_10_img/images/


    # Mention json path
    json_path=$ROCAL_DATA_PATH/rocal_data/coco/coco_10_img/annotations/coco_data.json

    # coco_reader.py
    # By default : cpu backend, NCHW format , fp32
    # Please pass annotation path in addition to other common args
    # Annotation must be a json file
    python"$ver" coco_reader.py \
        --image-dataset-path $data_dir \
        --json-path $json_path \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ caffe_reader -eq 1 ]]; then

    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/rocal_data/caffe/classification/

    # caffe_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python"$ver" caffe_reader.py \
        --image-dataset-path $data_dir \
        --classification \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################



####################################################################################################################################
if [[ caffe_reader -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/rocal_data/caffe/detection/

    # caffe_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python"$ver" caffe_reader.py \
        --image-dataset-path $data_dir \
        --no-classification \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################



####################################################################################################################################
if [[ caffe2_reader -eq 1 ]]; then

    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/rocal_data/caffe2/classification/

    # caffe2_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python"$ver" caffe2_reader.py \
        --image-dataset-path $data_dir \
        --classification \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ caffe2_reader -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/rocal_data/caffe2/detection/

    # caffe2_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python"$ver" caffe2_reader.py \
        --image-dataset-path $data_dir \
        --no-classification \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ tf_classification_reader -eq 1 ]]; then

    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/rocal_data/tf/classification/
    # tf_classification_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python"$ver" tf_classification_reader.py \
        --image-dataset-path $data_dir \
        --classification \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ tf_detection_reader -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/rocal_data/tf/detection/
    # tf_detection_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python"$ver" tf_detection_reader.py \
        --image-dataset-path $data_dir \
        --no-classification \
        --batch-size 100 \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ video_pipeline -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/rocal_data/video_and_sequence_samples/labelled_videos/
    # video_pipeline.py
    # By default : cpu backend, NCHW format , fp32

    python"$ver" video_pipeline.py \
        --video-path $data_dir \
        --$backend_arg \
        --batch-size 10 \
        --$display_arg \
        --$print_tensor_arg \
        --sequence-length 3 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
fi
####################################################################################################################################

####################################################################################################################################
if [[ web_dataset_reader -eq 1 ]]; then

    echo "Web Dataset Reader"
    echo $ROCAL_DATA_PATH
    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/rocal_data/web_dataset/tar_file/
    echo $data_dir
    echo $ROCAL_DATA_PATH
    echo "Web Dataset Reader 1"
    # Mention index path
    index_path=$ROCAL_DATA_PATH/rocal_data/web_dataset/idx_file/
    echo "Web Dataset Reader 2"
    # web_dataset_reader.py
    # By default : cpu backend, NCHW format , fp32
    echo "Web Dataset Reader 3"
    python"$ver" web_dataset_reader.py \
        --image-dataset-path $data_dir \
        --index-path "" \
        --classification \
        --batch-size $batch_size \
        --$display_arg \
        --$backend_arg \
        --NHWC \
        --local-rank 0 \
        --$print_tensor_arg \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_log.${CURRENTDATE}.txt
    echo "Web Dataset Reader 4"
fi
####################################################################################################################################