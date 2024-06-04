#!/bin/bash
cwd=$(pwd)
if [ -d build ];then 
    sudo rm -rf ./build/*
else
    mkdir build
fi
cd build || exit
cmake ..
make -j"$(nproc)"

if [[ $ROCAL_DATA_PATH == "" ]]
then 
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

# Path to inputs and outputs available in MIVisionX-data
image_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/images/
coco_detection_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/images/
tf_classification_path=${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path=${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe/detection/
caffe2_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
caffe2_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/detection/
mxnet_path=${ROCAL_DATA_PATH}/rocal_data/mxnet/
output_path=../rocal_unittest_output_folder_$(date +%Y-%m-%d_%H-%M-%S)/
golden_output_path=${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/

display=0
device=0
width=416 
height=416
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
        echo "Running HIP Backend..."
    else
        echo "Running HOST Backend..."
    fi
    for ((rgb=rgb_start;rgb<=rgb_end;rgb++))
    do 
        # FileSource Reader
        ./unit_tests 0 "$image_path" "${output_path}LensCorrection_${rgb_name[$rgb]}_${device_name}" $width $height 45 $device $rgb 0 $display
        ./unit_tests 0 "$image_path" "${output_path}Exposure_${rgb_name[$rgb]}_${device_name}" $width $height 46 $device $rgb 0 $display
        ./unit_tests 0 "$image_path" "${output_path}Flip_${rgb_name[$rgb]}_${device_name}" $width $height 47 $device $rgb 0 $display

        # FileSource Reader + partial decoder
        ./unit_tests 1 "$image_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}_FileReader_partial" $width $height 41 $device $rgb 0 $display
        ./unit_tests 1 "$image_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}_FileReader_partial" $width $height 42 $device $rgb 0 $display
        ./unit_tests 1 "$image_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}_FileReader_partial" $width $height 40 $device $rgb 0 $display

        # coco detection
        ./unit_tests 2 "$coco_detection_path" "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}" $width $height 33 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}Contrast_${rgb_name[$rgb]}_${device_name}" $width $height 34 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}Vignette_${rgb_name[$rgb]}_${device_name}" $width $height 38 $device $rgb 0 $display

        # coco detection + partial decoder
        ./unit_tests 3 "$coco_detection_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}_coco_partial" $width $height 41 $device $rgb 0 $display
        ./unit_tests 3 "$coco_detection_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}_coco_partial" $width $height 42 $device $rgb 0 $display
        ./unit_tests 3 "$coco_detection_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}_coco_partial" $width $height 40 $device $rgb 0 $display

        # tf classification
        ./unit_tests 4 "$tf_classification_path" "${output_path}Blend_${rgb_name[$rgb]}_${device_name}" $width $height 36 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}WarpAffine_${rgb_name[$rgb]}_${device_name}" $width $height 37 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}Blur_${rgb_name[$rgb]}_${device_name}" $width $height 35 $device $rgb 0 $display

        # tf detection
        ./unit_tests 5 "$tf_detection_path" "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}" $width $height 40 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}ColorTemp_${rgb_name[$rgb]}_${device_name}" $width $height 43 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}Fog_${rgb_name[$rgb]}_${device_name}" $width $height 44 $device $rgb 0 $display

        # caffe classification
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Rotate_${rgb_name[$rgb]}_${device_name}" $width $height 31 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Brightness_${rgb_name[$rgb]}_${device_name}" $width $height 32 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Hue_${rgb_name[$rgb]}_${device_name}" $width $height 48 $device $rgb 0 $display

        # caffe detection
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Saturation_${rgb_name[$rgb]}_${device_name}" $width $height 49 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}ColorTwist_${rgb_name[$rgb]}_${device_name}" $width $height 50 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Rain_${rgb_name[$rgb]}_${device_name}" $width $height 42 $device $rgb 0 $display

        # caffe2 classification
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}CropCenter_${rgb_name[$rgb]}_${device_name}" $width $height 52 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}ResizeCropMirror_${rgb_name[$rgb]}_${device_name}" $width $height 53 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}Snow_${rgb_name[$rgb]}_${device_name}" $width $height 41 $device $rgb 0 $display

        # caffe2 detection
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}FishEye_${rgb_name[$rgb]}_${device_name}" $width $height 10 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}Pixelate_${rgb_name[$rgb]}_${device_name}" $width $height 19 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}CropCenterCMN_${rgb_name[$rgb]}_${device_name}" $width $height 55 $device $rgb 0 $display

        # mxnet 
        ./unit_tests 11 "$mxnet_path" "${output_path}Jitter_${rgb_name[$rgb]}_${device_name}" $width $height 39 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}Pixelate_${rgb_name[$rgb]}_${device_name}" $width $height 19 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 25 $device $rgb 0 $display

        # CMN 
        ./unit_tests 0 "$image_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_FileReader" $width $height 25 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_coco" $width $height 25 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfClassification" $width $height 25 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfDetection" $width $height 25 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeClassification" $width $height 25 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeDetection" $width $height 25 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Classification" $width $height 25 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Detection" $width $height 25 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 25 $device $rgb 0 $display

        # crop
        ./unit_tests 0 "$image_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_FileReader" $width $height 51 $device $rgb 0 $display
        ./unit_tests 2 "$coco_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_coco" $width $height 51 $device $rgb 0 $display
        ./unit_tests 4 "$tf_classification_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfClassification" $width $height 51 $device $rgb 0 $display
        ./unit_tests 5 "$tf_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfDetection" $width $height 51 $device $rgb 0 $display
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeClassification" $width $height 51 $device $rgb 0 $display
        ./unit_tests 7 "$caffe_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeDetection" $width $height 51 $device $rgb 0 $display
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Classification" $width $height 51 $device $rgb 0 $display
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Detection" $width $height 51 $device $rgb 0 $display
        ./unit_tests 11 "$mxnet_path" "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_mxnet" $width $height 51 $device $rgb 0 $display

        # resize
        # Last two parameters are interpolation type and scaling mode
        ./unit_tests 0 "$image_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_default_FileReader" $width $height 0 $device $rgb 0 $display 1 0 
        ./unit_tests 2 "$coco_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_stretch_coco" $width $height 0 $device $rgb 0 $display 1 1
        ./unit_tests 4 "$tf_classification_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notsmaller_tfClassification" $width $height 0 $device $rgb 0 $display 1 2
        ./unit_tests 5 "$tf_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notlarger_tfDetection" $width $height 0 $device $rgb 0 $display 1 3
        ./unit_tests 6 "$caffe_classification_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bicubic_default_caffeClassification" $width $height 0 $device $rgb 0 $display 2 0
        # ./unit_tests 7 "$caffe_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_nearestneighbor_default_caffeDetection" $width $height 0 $device $rgb 0 $display 0 0
        ./unit_tests 8 "$caffe2_classification_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_lanczos_default_caffe2Classification" $width $height 0 $device $rgb 0 $display 3 0
        ./unit_tests 9 "$caffe2_detection_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_triangular_default_caffe2Detection" $width $height 0 $device $rgb 0 $display 5 0
        ./unit_tests 11 "$mxnet_path" "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_gaussian_default_mxnet" $width $height 0 $device $rgb 0 $display 4 0 

    done
done

pwd

# Run python script to compare rocAL outputs with golden ouptuts
python3 "$cwd"/pixel_comparison/image_comparison.py "$golden_output_path" "$output_path"
