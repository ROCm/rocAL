# rocAL Serialization Test

This test demonstrates the rocAL pipeline serialization functionality using a simple pipeline with:
- JPEG file source (`rocalJpegFileSource`)
- Label reader (`rocalCreateLabelReader`)
- Brightness augmentation (`rocalBrightness`)

## Purpose

The test validates the `rocalSerialize` and `rocalGetSerializedString` APIs by:
1. Creating a rocAL pipeline with the specified components
2. Serializing the pipeline to get the serialized string size
3. Retrieving and printing the serialized pipeline string
4. Running a few iterations to verify the pipeline works correctly

## Pre-requisites

* Ubuntu Linux, version `22.04` or later
* rocAL library
* [OpenCV 4.0+](https://github.com/opencv/opencv/releases/tag/4.0.0)
* ROCm Performance Primitives ([RPP](https://github.com/ROCm/rpp))

## Building

```bash
cd tests/cpp_api/serialization_test
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./serialization_test <image_dataset_folder> [processing_device]
```

### Parameters:
- `image_dataset_folder`: Path to folder containing JPEG images (required)
- `processing_device`: 0 for CPU, 1 for GPU (optional, default: 0)

### Example:
```bash
# Using CPU processing
./serialization_test /path/to/images 0

# Using GPU processing
./serialization_test /path/to/images 1
```

## Expected Output

The test will:
1. Create the rocAL pipeline
2. Display pipeline information (dimensions, augmentation count)
3. Serialize the pipeline and show the serialized string size
4. Print the complete serialized pipeline string
5. Run a few iterations to process images and display image names with labels
6. Report successful completion

## Test Data

You can use the sample data provided in the rocAL repository:
```bash
./serialization_test ../../../data/images/AMD-tinyDataSet 0
