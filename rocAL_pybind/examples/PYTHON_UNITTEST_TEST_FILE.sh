#!/bin/bash
cwd=$(pwd)

if [[ $ROCAL_DATA_PATH == "" ]]
then 
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

# Path to inputs and outputs available in MIVisionX-data
image_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
coco_detection_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/train_10images_2017/
coco_json_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/annotations/instances_train2017.json
tf_classification_path=${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path=${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe/detection/
caffe2_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
caffe2_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/detection/
mxnet_path=${ROCAL_DATA_PATH}/rocal_data/mxnet/
output_path=./rocal_python_unittest_output_folder_$(date +%Y-%m-%d_%H-%M-%S)/
golden_output_path=${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/

display=0
batch_size=2
device=0
width=640 
height=480
device_name="host"
rgb_name=("gray" "rgb")
rgb=1
dev_start=0
dev_end=1
rgb_start=0
rgb_end=1

if [ "$#" -gt 0 ]; then 
    if [ "$1" -eq 0 ]; then # For only HOST backend
        dev_start=0
        dev_end=0
    elif [ "$1" -eq 1 ]; then # For only HIP backend
        dev_start=1
        dev_end=1
    elif [ "$1" -eq 2 ]; then # For both HOST and HIP backend
        dev_start=0
        dev_end=1
    fi
fi

if [ "$#" -gt 1 ]; then
    if [ "$2" -eq 0 ]; then # For only Greyscale inputs
        rgb_start=0
        rgb_end=0
    elif [ "$2" -eq 1 ]; then # For only RGB inputs
        rgb_start=1
        rgb_end=1
    elif [ "$2" -eq 2 ]; then # For both RGB and Greyscale inputs
        rgb_start=0
        rgb_end=1
    fi
fi

mkdir "$output_path"

for ((device=dev_start;device<=dev_end;device++))
do 
    if [ $device -eq 1 ]
    then 
        device_name="hip"
        backend_arg=rocal-gpu
        echo "Running HIP Backend..."
    else
        backend_arg=no-rocal-gpu
        echo "Running HOST Backend..."
    fi
    for ((rgb=rgb_start;rgb<=rgb_end;rgb++))
    do 
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$image_path" --augmentation-name lens_correction --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}LensCorrection_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$image_path" --augmentation-name exposure --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Exposure_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$image_path" --augmentation-name flip --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Flip_${rgb_name[$rgb]}_${device_name}"

        # coco detection
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name gamma_correction --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name contrast --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Contrast_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name vignette --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Vignette_${rgb_name[$rgb]}_${device_name}"

        # # tf classification
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name blend --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Blend_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name warp_affine --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}WarpAffine_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name blur --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Blur_${rgb_name[$rgb]}_${device_name}"

        # tf detection
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name snp_noise --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name color_temp --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ColorTemp_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name fog --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Fog_${rgb_name[$rgb]}_${device_name}"

        # caffe classification
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name rotate --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Rotate_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name brightness --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Brightness_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name hue --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Hue_${rgb_name[$rgb]}_${device_name}"

        # caffe detection
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name saturation --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Saturation_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name color_twist --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ColorTwist_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name rain --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Rain_${rgb_name[$rgb]}_${device_name}"

        # caffe2 classification
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name center_crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropCenter_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name resize_crop_mirror --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ResizeCropMirror_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name snow --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Snow_${rgb_name[$rgb]}_${device_name}"

        # caffe2 detection
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name fish_eye --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}FishEye_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name pixelate --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Pixelate_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name center_crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropCenter_${rgb_name[$rgb]}_${device_name}_cmn"

        # mxnet
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name jitter --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Jitter_${rgb_name[$rgb]}_${device_name}"        
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name resize_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ResizeMirrorNormalize_${rgb_name[$rgb]}_${device_name}"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet"

        # CMN 
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$image_path" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_FileReader"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_coco"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfClassification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfDetection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeClassification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeDetection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Classification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Detection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet"

        # crop
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$image_path" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_FileReader"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_coco"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfClassification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfDetection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeClassification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeDetection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Classification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Detection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_mxnet"

        # resize
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$image_path" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_default_FileReader"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 1 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_stretch_coco"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 2 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notsmaller_tfClassification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 3 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notlarger_tfDetection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 2 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bicubic_default_caffeClassification"
        # python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 0 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_nearestneighbor_default_caffeDetection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 3 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_lanczos_default_caffe2Classification"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 5 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_triangular_default_caffe2Detection"
        python"$ver" rocAL_api_python_unittest.py --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 4 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_gaussian_default_mxnet"

    done
done

pwd

# Run python script to compare rocAL outputs with golden ouptuts
python3 "$cwd"/image_comparison.py "$golden_output_path" "$output_path"
