/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include "rocal_api_types.h"
#include "rocal_api.h"
#include "rocal_api_tensor.h"
#include "rocal_api_parameters.h"
#include "rocal_api_data_loaders.h"
#include "rocal_api_augmentation.h"
#include "rocal_api_data_transfer.h"
#include "rocal_api_info.h"
namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");
namespace pybind11 {
namespace detail {
constexpr int NPY_FLOAT16 = 23;

template <>
struct npy_format_descriptor<float16> {
    static constexpr auto name = _("float16");
    static pybind11::dtype dtype() {
        handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
        return reinterpret_borrow<pybind11::dtype>(ptr);
    }
};
}  // namespace detail
}  // namespace pybind11
namespace rocal {
using namespace pybind11::literals;  // NOLINT
// PYBIND11_MODULE(rocal_backend_impl, m) {
static void *ctypes_void_ptr(const py::object &object) {
    auto ptr_as_int = getattr(object, "value", py::none());
    if (ptr_as_int.is_none()) {
        return nullptr;
    }
    void *ptr = PyLong_AsVoidPtr(ptr_as_int.ptr());
    return ptr;
}

py::object wrapper_image_name_length(RocalContext context, py::array_t<int> array) {
    auto buf = array.request();
    int *ptr = static_cast<int *>(buf.ptr);
    // call pure C++ function
    int length = rocalGetImageNameLen(context, ptr);
    return py::cast(length);
}

py::object wrapper_image_name(RocalContext context, int array_len) {
    py::array_t<char> array;
    auto buf = array.request();
    char *ptr = static_cast<char *>(buf.ptr);
    ptr = static_cast<char *>(calloc(array_len, sizeof(char)));
    // call pure C++ function
    rocalGetImageName(context, ptr);
    std::string s(ptr);
    free(ptr);
    return py::bytes(s);
}

py::object wrapper_copy_to_tensor(RocalContext context, py::object p,
                                  RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type, float multiplier0,
                                  float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                                  bool reverse_channels, RocalOutputMemType output_mem_type, uint max_roi_height, uint max_roi_width) {
    auto ptr = ctypes_void_ptr(p);
    // call pure C++ function
    int status = rocalToTensor(context, ptr, tensor_format, tensor_output_type, multiplier0,
                               multiplier1, multiplier2, offset0, offset1, offset2,
                               reverse_channels, output_mem_type, max_roi_height, max_roi_width);
    return py::cast<py::none>(Py_None);
}

std::unordered_map<int, std::string> rocalToPybindLayout = {
    {0, "NHWC"},
    {1, "NCHW"},
    {2, "NFHWC"},
    {3, "NFCHW"},
};

std::unordered_map<int, std::string> rocalToPybindOutputDtype = {
    {0, "float32"},
    {1, "float16"},
    {2, "uint8"},
    {3, "int8"},
};

PYBIND11_MODULE(rocal_pybind, m) {
    m.doc() = "Python bindings for the C++ portions of ROCAL";
    // rocal_api.h
    m.def("rocalCreate", &rocalCreate, "Creates context with the arguments sent and returns it",
          py::return_value_policy::reference,
          py::arg("batch_size"),
          py::arg("affinity"),
          py::arg("gpu_id") = 0,
          py::arg("cpu_thread_count") = 1,
          py::arg("prefetch_queue_depth") = 3,
          py::arg("output_data_type") = 0);
    m.def("rocalVerify", &rocalVerify);
    m.def("rocalRun", &rocalRun, py::return_value_policy::reference);
    m.def("rocalRelease", &rocalRelease, py::return_value_policy::reference);
    // rocal_api_types.h
    py::class_<TimingInfo>(m, "TimingInfo")
        .def_readwrite("load_time", &TimingInfo::load_time)
        .def_readwrite("decode_time", &TimingInfo::decode_time)
        .def_readwrite("process_time", &TimingInfo::process_time)
        .def_readwrite("transfer_time", &TimingInfo::transfer_time);
    py::class_<rocalTensor>(m, "rocalTensor")
        .def(
            "max_shape",
            [](rocalTensor &output_tensor) {
                return output_tensor.shape();
            },
            R"code(
                Returns a tensor buffer's shape.
                )code")
        .def(
            "batch_size",
            [](rocalTensor &output_tensor) {
                return output_tensor.dims().at(0);
            },
            R"code(
                Returns a tensor batch size.
                )code")
        .def(
            "layout", [](rocalTensor &output_tensor) {
                return rocalToPybindLayout[(int)output_tensor.layout()];
            },
            R"code(
                Returns layout of tensor.
                )code")
        .def(
            "dtype", [](rocalTensor &output_tensor) {
                return rocalToPybindOutputDtype[(int)output_tensor.data_type()];
            },
            R"code(
                Returns dtype of tensor.
                )code")
        .def(
            "dimensions", [](rocalTensor &output_tensor) {
                return output_tensor.dims();
            },
            R"code(
                Returns dims of tensor.
                )code")
        .def(
            "roi_dims_size", [](rocalTensor &output_tensor) {
                return output_tensor.get_roi_dims_size();
            },
            R"code(
                Returns the number of dims for ROI data
                )code")
        .def(
            "copy_roi", [](rocalTensor &output_tensor, py::array array) {
                auto buf = array.request();
                output_tensor.copy_roi(static_cast<void *>(buf.ptr));
            },
            R"code(
                Copies the ROI data to numpy arrays.
                )code")
        .def(
            "copy_data", [](rocalTensor &output_tensor, py::object p, RocalOutputMemType external_mem_type) {
                auto ptr = ctypes_void_ptr(p);
                output_tensor.copy_data(static_cast<void *>(ptr), external_mem_type);
            },
            R"code(
                Copies the ring buffer data to python buffer pointers.
                )code")
        .def(
            "copy_data", [](rocalTensor &output_tensor, py::array array) {
                auto buf = array.request();
                output_tensor.copy_data(static_cast<void *>(buf.ptr), RocalOutputMemType::ROCAL_MEMCPY_HOST);
            },
            py::return_value_policy::reference,
            R"code(
                Copies the ring buffer data to numpy arrays.
                )code")
        .def(
            "copy_data", [](rocalTensor &output_tensor, long array) {
                output_tensor.copy_data((void *)array, RocalOutputMemType::ROCAL_MEMCPY_GPU);
            },
            py::return_value_policy::reference,
            R"code(
                Copies the ring buffer data to cupy arrays.
                )code")
        .def(
            "at", [](rocalTensor &output_tensor, uint idx) {
                std::vector<size_t> stride_per_sample(output_tensor.strides());
                stride_per_sample.erase(stride_per_sample.begin());
                std::vector<size_t> dims(output_tensor.dims());
                dims.erase(dims.begin());
                py::array numpy_array;
                switch (output_tensor.data_type()) {
                    case RocalTensorOutputType::ROCAL_UINT8:
                        numpy_array = py::array(py::buffer_info(
                            (static_cast<unsigned char *>(output_tensor.buffer())) + idx * (output_tensor.strides()[0] / sizeof(unsigned char)),
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor.num_of_dims() - 1,
                            dims,
                            stride_per_sample));
                        break;
                    case RocalTensorOutputType::ROCAL_FP32:
                        numpy_array = py::array(py::buffer_info(
                            (static_cast<float *>(output_tensor.buffer())) + idx * (output_tensor.strides()[0] / sizeof(float)),
                            sizeof(float),
                            py::format_descriptor<float>::format(),
                            output_tensor.num_of_dims() - 1,
                            dims,
                            stride_per_sample));
                        break;
                    default:
                        throw py::type_error("Unknown rocAL data type");
                }
                return numpy_array;
            },
            "idx"_a,
            R"code(
                Returns a rocal tensor at given position `idx` in the rocalTensorlist.
                )code",
            py::keep_alive<0, 1>());
    py::class_<rocalTensorList>(m, "rocalTensorList")
        .def(
            "__getitem__",
            [](rocalTensorList &output_tensor_list, uint idx) {
                return output_tensor_list.at(idx);
            },
            R"code(
                Returns a tensor at given position in the list.
                )code")

        .def(
            "at",
            [](rocalTensorList &output_tensor_list, uint idx) {
                auto output_tensor = output_tensor_list.at(idx);
                py::array numpy_array;
                switch (output_tensor->data_type()) {
                    case RocalTensorOutputType::ROCAL_UINT8:
                        numpy_array = py::array(py::buffer_info(
                            static_cast<unsigned char *>(output_tensor->buffer()),
                            sizeof(unsigned char),
                            py::format_descriptor<unsigned char>::format(),
                            output_tensor->num_of_dims(),
                            output_tensor->dims(),
                            output_tensor->strides()));
                        break;
                    case RocalTensorOutputType::ROCAL_FP32:
                        numpy_array = py::array(py::buffer_info(
                            static_cast<float *>(output_tensor->buffer()),
                            sizeof(float),
                            py::format_descriptor<float>::format(),
                            output_tensor->num_of_dims(),
                            output_tensor->dims(),
                            output_tensor->strides()));
                        break;
                    default:
                        throw py::type_error("Unknown rocAL data type");
                }
                return numpy_array;
            },
            "idx"_a,
            R"code(
                Returns a rocal tensor at given position `i` in the rocalTensorlist.
                )code",
            py::keep_alive<0, 1>());

    py::module types_m = m.def_submodule("types");
    types_m.doc() = "Datatypes and options used by ROCAL";
    py::enum_<RocalStatus>(types_m, "RocalStatus", "Status info")
        .value("OK", ROCAL_OK)
        .value("CONTEXT_INVALID", ROCAL_CONTEXT_INVALID)
        .value("RUNTIME_ERROR", ROCAL_RUNTIME_ERROR)
        .value("UPDATE_PARAMETER_FAILED", ROCAL_UPDATE_PARAMETER_FAILED)
        .value("INVALID_PARAMETER_TYPE", ROCAL_INVALID_PARAMETER_TYPE)
        .export_values();
    py::enum_<RocalProcessMode>(types_m, "RocalProcessMode", "Processing mode")
        .value("GPU", ROCAL_PROCESS_GPU)
        .value("CPU", ROCAL_PROCESS_CPU)
        .export_values();
    py::enum_<RocalTensorOutputType>(types_m, "RocalTensorOutputType", "Tensor types")
        .value("FLOAT", ROCAL_FP32)
        .value("FLOAT16", ROCAL_FP16)
        .value("UINT8", ROCAL_UINT8)
        .export_values();
    py::enum_<RocalOutputMemType>(types_m, "RocalOutputMemType", "Output memory types")
        .value("HOST_MEMORY", ROCAL_MEMCPY_HOST)
        .value("DEVICE_MEMORY", ROCAL_MEMCPY_GPU)
        .value("PINNED_MEMORY", ROCAL_MEMCPY_PINNED)
        .export_values();
    py::enum_<RocalResizeScalingMode>(types_m, "RocalResizeScalingMode", "Decode size policies")
        .value("SCALING_MODE_DEFAULT", ROCAL_SCALING_MODE_DEFAULT)
        .value("SCALING_MODE_STRETCH", ROCAL_SCALING_MODE_STRETCH)
        .value("SCALING_MODE_NOT_SMALLER", ROCAL_SCALING_MODE_NOT_SMALLER)
        .value("SCALING_MODE_NOT_LARGER", ROCAL_SCALING_MODE_NOT_LARGER)
        .value("SCALING_MODE_MIN_MAX", ROCAL_SCALING_MODE_MIN_MAX)
        .export_values();
    py::enum_<RocalResizeInterpolationType>(types_m, "RocalResizeInterpolationType", "Decode size policies")
        .value("NEAREST_NEIGHBOR_INTERPOLATION", ROCAL_NEAREST_NEIGHBOR_INTERPOLATION)
        .value("LINEAR_INTERPOLATION", ROCAL_LINEAR_INTERPOLATION)
        .value("CUBIC_INTERPOLATION", ROCAL_CUBIC_INTERPOLATION)
        .value("LANCZOS_INTERPOLATION", ROCAL_LANCZOS_INTERPOLATION)
        .value("GAUSSIAN_INTERPOLATION", ROCAL_GAUSSIAN_INTERPOLATION)
        .value("TRIANGULAR_INTERPOLATION", ROCAL_TRIANGULAR_INTERPOLATION)
        .export_values();
    py::enum_<RocalImageSizeEvaluationPolicy>(types_m, "RocalImageSizeEvaluationPolicy", "Decode size policies")
        .value("MAX_SIZE", ROCAL_USE_MAX_SIZE)
        .value("USER_GIVEN_SIZE", ROCAL_USE_USER_GIVEN_SIZE)
        .value("MOST_FREQUENT_SIZE", ROCAL_USE_MOST_FREQUENT_SIZE)
        .value("MAX_SIZE_ORIG", ROCAL_USE_MAX_SIZE_RESTRICTED)
        .value("USER_GIVEN_SIZE_ORIG", ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED)
        .export_values();
    py::enum_<RocalImageColor>(types_m, "RocalImageColor", "Image type")
        .value("RGB", ROCAL_COLOR_RGB24)
        .value("BGR", ROCAL_COLOR_BGR24)
        .value("GRAY", ROCAL_COLOR_U8)
        .value("RGB_PLANAR", ROCAL_COLOR_RGB_PLANAR)
        .export_values();
    py::enum_<RocalTensorLayout>(types_m, "RocalTensorLayout", "Tensor layout type")
        .value("NHWC", ROCAL_NHWC)
        .value("NCHW", ROCAL_NCHW)
        .value("NFHWC", ROCAL_NFHWC)
        .value("NFCHW", ROCAL_NFCHW)
        .export_values();
    py::enum_<RocalDecodeDevice>(types_m, "RocalDecodeDevice", "Decode device type")
        .value("HARDWARE_DECODE", ROCAL_HW_DECODE)
        .value("SOFTWARE_DECODE", ROCAL_SW_DECODE)
        .export_values();
    py::enum_<RocalDecoderType>(types_m, "RocalDecoderType", "Rocal Decoder Type")
        .value("DECODER_TJPEG", ROCAL_DECODER_TJPEG)
        .value("DECODER_OPENCV", ROCAL_DECODER_OPENCV)
        .value("DECODER_HW_JEPG", ROCAL_DECODER_HW_JPEG)
        .value("DECODER_VIDEO_FFMPEG_SW", ROCAL_DECODER_VIDEO_FFMPEG_SW)
        .value("DECODER_VIDEO_FFMPEG_HW", ROCAL_DECODER_VIDEO_FFMPEG_HW)
        .export_values();
    // rocal_api_info.h
    m.def("getRemainingImages", &rocalGetRemainingImages);
    m.def("getImageName", &wrapper_image_name);
    m.def("getImageId", [](RocalContext context, py::array_t<int> array) {
        auto buf = array.request();
        int *ptr = static_cast<int *>(buf.ptr);
        return rocalGetImageId(context, ptr);
    });
    m.def("getImageNameLen", &wrapper_image_name_length);
    m.def("getStatus", &rocalGetStatus);
    m.def("setOutputs", &rocalSetOutputs);
    m.def("tfReader", &rocalCreateTFReader, py::return_value_policy::reference);
    m.def("tfReaderDetection", &rocalCreateTFReaderDetection, py::return_value_policy::reference);
    m.def("caffeReader", &rocalCreateCaffeLMDBLabelReader, py::return_value_policy::reference);
    m.def("caffe2Reader", &rocalCreateCaffe2LMDBLabelReader, py::return_value_policy::reference);
    m.def("caffeReaderDetection", &rocalCreateCaffeLMDBReaderDetection, py::return_value_policy::reference);
    m.def("caffe2ReaderDetection", &rocalCreateCaffe2LMDBReaderDetection, py::return_value_policy::reference);
    m.def("mxnetReader", &rocalCreateMXNetReader, py::return_value_policy::reference);
    m.def("isEmpty", &rocalIsEmpty);
    m.def("getStatus", rocalGetStatus);
    m.def("rocalGetErrorMessage", &rocalGetErrorMessage);
    m.def("getTimingInfo", &rocalGetTimingInfo);
    m.def("labelReader", &rocalCreateLabelReader, py::return_value_policy::reference);
    m.def("cocoReader", &rocalCreateCOCOReader, py::return_value_policy::reference);
    // rocal_api_meta_data.h
    m.def("randomBBoxCrop", &rocalRandomBBoxCrop);
    m.def("boxEncoder", &rocalBoxEncoder);
    // m.def("BoxIOUMatcher", &rocalBoxIOUMatcher);  // Will be enabled when IOU matcher changes are introduced in C++
    m.def("getImgSizes", [](RocalContext context, py::array_t<int> array) {
        auto buf = array.request();
        int *ptr = static_cast<int *>(buf.ptr);
        rocalGetImageSizes(context, ptr);
    });
    m.def("getROIImgSizes", [](RocalContext context, py::array_t<int> array) {
        auto buf = array.request();
        int *ptr = static_cast<int *>(buf.ptr);
        rocalGetROIImageSizes(context, ptr);
    });
    // rocal_api_parameter.h
    m.def("setSeed", &rocalSetSeed);
    m.def("getSeed", &rocalGetSeed);
    m.def("createIntUniformRand", &rocalCreateIntUniformRand, py::return_value_policy::reference);
    m.def("createFloatUniformRand", &rocalCreateFloatUniformRand, py::return_value_policy::reference);
    m.def(
        "createIntRand", [](std::vector<int> values, std::vector<double> frequencies) {
            return rocalCreateIntRand(values.data(), frequencies.data(), values.size());
        },
        py::return_value_policy::reference);
    m.def("createFloatRand", &rocalCreateFloatRand, py::return_value_policy::reference);
    m.def("createIntParameter", &rocalCreateIntParameter, py::return_value_policy::reference);
    m.def("createFloatParameter", &rocalCreateFloatParameter, py::return_value_policy::reference);
    m.def("updateIntRand", &rocalUpdateIntUniformRand);
    m.def("updateFloatRand", &rocalUpdateFloatUniformRand);
    m.def("updateIntParameter", &rocalUpdateIntParameter);
    m.def("updateFloatParameter", &rocalUpdateFloatParameter);
    m.def("getIntValue", &rocalGetIntValue);
    m.def("getFloatValue", &rocalGetFloatValue);
    // rocal_api_data_transfer.h
    m.def("rocalToTensor", &wrapper_copy_to_tensor);
    m.def("getOutputTensors", [](RocalContext context) {
        rocalTensorList *output_tensor_list = rocalGetOutputTensors(context);
        py::list list;
        unsigned int size_of_tensor_list = output_tensor_list->size();
        for (uint i = 0; i < size_of_tensor_list; i++)
            list.append(output_tensor_list->at(i));
        return list;
    });
    m.def("getBoundingBoxCount", &rocalGetBoundingBoxCount);
    m.def("getImageLabels", [](RocalContext context) {
        rocalTensorList *labels = rocalGetImageLabels(context);
        return py::array(py::buffer_info(
            static_cast<int *>(labels->at(0)->buffer()),
            sizeof(int),
            py::format_descriptor<int>::format(),
            1,
            {labels->size()},
            {sizeof(int)}));
    });
    m.def("getBoundingBoxLabels", [](RocalContext context) {
        rocalTensorList *labels = rocalGetBoundingBoxLabel(context);
        py::list labels_list;
        py::array_t<int> labels_array;
        for (int i = 0; i < labels->size(); i++) {
            int *labels_buffer = static_cast<int *>(labels->at(i)->buffer());
            labels_array = py::array(py::buffer_info(
                static_cast<int *>(labels->at(i)->buffer()),
                sizeof(int),
                py::format_descriptor<int>::format(),
                1,
                {labels->at(i)->dims().at(0)},
                {sizeof(int)}));
            labels_list.append(labels_array);
        }
        return labels_list;
    });
    m.def("getBoundingBoxCords", [](RocalContext context) {
        rocalTensorList *boxes = rocalGetBoundingBoxCords(context);
        py::list boxes_list;
        py::array_t<float> boxes_array;
        for (int i = 0; i < boxes->size(); i++) {
            float *box_buffer = static_cast<float *>(boxes->at(i)->buffer());
            boxes_array = py::array(py::buffer_info(
                static_cast<float *>(boxes->at(i)->buffer()),
                sizeof(float),
                py::format_descriptor<float>::format(),
                1,
                {boxes->at(i)->dims().at(0) * 4},
                {sizeof(float)}));
            boxes_list.append(boxes_array);
        }
        return boxes_list;
    });
    m.def("getMaskCount", [](RocalContext context, py::array_t<int> array) {
        auto buf = array.mutable_data();
        unsigned count = rocalGetMaskCount(context, buf);  // total number of polygons in complete batch
        return count;
    });
    m.def("getMaskCoordinates", [](RocalContext context, py::array_t<int> polygon_size, py::array_t<int> mask_count) {
        auto buf = polygon_size.request();
        int *polygon_size_ptr = static_cast<int *>(buf.ptr);
        // call pure C++ function
        rocalTensorList *mask_data = rocalGetMaskCoordinates(context, polygon_size_ptr);
        rocalTensorList *bbox_labels = rocalGetBoundingBoxLabel(context);
        py::list complete_list;
        int poly_cnt = 0;
        int prev_object_cnt = 0;
        auto mask_count_buf = mask_count.request();
        int *mask_count_ptr = static_cast<int *>(mask_count_buf.ptr);
        for (int i = 0; i < bbox_labels->size(); i++) {  // For each image in a batch, parse through the mask metadata buffers and convert them to polygons format
            float *mask_buffer = static_cast<float *>(mask_data->at(i)->buffer());
            py::list poly_batch_list;
            for (unsigned j = prev_object_cnt; j < bbox_labels->at(i)->dims().at(0) + prev_object_cnt; j++) {
                py::list polygons_buffer;
                for (int k = 0; k < mask_count_ptr[j]; k++) {
                    py::list coords_buffer;
                    for (int l = 0; l < polygon_size_ptr[poly_cnt]; l++)
                        coords_buffer.append(mask_buffer[l]);
                    mask_buffer += polygon_size_ptr[poly_cnt++];
                    polygons_buffer.append(coords_buffer);
                }
                poly_batch_list.append(polygons_buffer);
            }
            prev_object_cnt += bbox_labels->at(i)->dims().at(0);
            complete_list.append(poly_batch_list);
        }
        return complete_list;
    });
    // Will be enabled when IOU matcher changes are introduced in C++
    // m.def("getMatchedIndices", [](RocalContext context) {
    //     rocalTensorList *matches = rocalGetMatchedIndices(context);
    //     return py::array(py::buffer_info(
    //                     (int *)(matches->at(0)->buffer()),
    //                     sizeof(int),
    //                     py::format_descriptor<int>::format(),
    //                     1,
    //                     {matches->size() * 120087},
    //                     {sizeof(int) }));
    // }, py::return_value_policy::reference);
    m.def("rocalGetEncodedBoxesAndLables", [](RocalContext context, uint batch_size, uint num_anchors) {
        auto vec_pair_labels_boxes = rocalGetEncodedBoxesAndLables(context, batch_size * num_anchors);
        auto labels_buf_ptr = static_cast<int *>(vec_pair_labels_boxes[0]->at(0)->buffer());
        auto bboxes_buf_ptr = static_cast<float *>(vec_pair_labels_boxes[1]->at(0)->buffer());

        py::array_t<int> labels_array = py::array_t<int>(py::buffer_info(
            labels_buf_ptr,
            sizeof(int),
            py::format_descriptor<int>::format(),
            2,
            {batch_size, num_anchors},
            {num_anchors * sizeof(int), sizeof(int)}));

        py::array_t<float> bboxes_array = py::array_t<float>(py::buffer_info(
            bboxes_buf_ptr,
            sizeof(float),
            py::format_descriptor<float>::format(),
            1,
            {batch_size * num_anchors * 4},
            {sizeof(float)}));
        return std::make_pair(labels_array, bboxes_array);
    });
    // rocal_api_data_loaders.h
    m.def("cocoImageDecoderSlice", &rocalJpegCOCOFileSourcePartial, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("cocoImageDecoderSliceShard", &rocalJpegCOCOFileSourcePartialSingleShard, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("imageDecoder", &rocalJpegFileSource, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("imageDecoderShard", &rocalJpegFileSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
          py::return_value_policy::reference);
    m.def("cocoImageDecoder", &rocalJpegCOCOFileSource, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("cocoImageDecoderShard", &rocalJpegCOCOFileSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
          py::return_value_policy::reference);
    m.def("tfImageDecoder", &rocalJpegTFRecordSource, "Reads file from the source given and decodes it according to the policy only for TFRecords",
          py::return_value_policy::reference);
    m.def("caffeImageDecoder", &rocalJpegCaffeLMDBRecordSource, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("caffeImageDecoderShard", &rocalJpegCaffeLMDBRecordSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
          py::return_value_policy::reference);
    m.def("caffeImageDecoderPartialShard", &rocalJpegCaffeLMDBRecordSourcePartialSingleShard, "Reads file from the source given and partially decodes it according to the shard id and number of shards",
          py::return_value_policy::reference);
    m.def("caffe2ImageDecoder", &rocalJpegCaffe2LMDBRecordSource, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("caffe2ImageDecoderShard", &rocalJpegCaffe2LMDBRecordSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
          py::return_value_policy::reference);
    m.def("caffe2ImageDecoderPartialShard", &rocalJpegCaffe2LMDBRecordSourcePartialSingleShard, "Reads file from the source given and partially decodes it according to the shard id and number of shards",
          py::return_value_policy::reference);
    m.def("fusedDecoderCrop", &rocalFusedJpegCrop, "Reads file from the source and decodes them partially to output random crops",
          py::return_value_policy::reference);
    m.def("fusedDecoderCropShard", &rocalFusedJpegCropSingleShard, "Reads file from the source and decodes them partially to output random crops",
          py::return_value_policy::reference);
    m.def("tfImageDecoderRaw", &rocalRawTFRecordSource, "Reads file from the source given and decodes it according to the policy only for TFRecords",
          py::return_value_policy::reference);
    m.def("cifar10Decoder", &rocalRawCIFAR10Source, "Reads file from the source given and decodes it according to the policy",
          py::return_value_policy::reference);
    m.def("videoDecoder", &rocalVideoFileSource, "Reads videos from the source given and decodes it according to the policy only for videos as inputs",
          py::return_value_policy::reference);
    m.def("videoDecoderResize", &rocalVideoFileResize, "Reads videos from the source given and decodes it according to the policy only for videos as inputs. Resizes the decoded frames to the dest width and height.",
          py::return_value_policy::reference);
    m.def("sequenceReader", &rocalSequenceReader, "Creates JPEG image reader and decoder. Reads [Frames] sequences from a directory representing a collection of streams.",
          py::return_value_policy::reference);
    m.def("mxnetDecoder", &rocalMXNetRecordSourceSingleShard, "Reads file from the source given and decodes it according to the policy only for mxnet records",
          py::return_value_policy::reference);
    m.def("rocalResetLoaders", &rocalResetLoaders);
    m.def("videoMetaDataReader", &rocalCreateVideoLabelReader, py::return_value_policy::reference);
    // rocal_api_augmentation.h
    m.def("ssdRandomCrop", &rocalSSDRandomCrop,
          py::return_value_policy::reference);
    m.def("resize", &rocalResize,
          py::return_value_policy::reference);
    m.def("resizeMirrorNormalize", &rocalResizeMirrorNormalize,
          py::return_value_policy::reference);
    m.def("resizeCropMirrorFixed", &rocalResizeCropMirrorFixed,
          py::return_value_policy::reference);
    m.def("cropResize", &rocalCropResize,
          py::return_value_policy::reference);
    m.def("copy", &rocalCopy,
          py::return_value_policy::reference);
    m.def("nop", &rocalNop,
          py::return_value_policy::reference);
    m.def("colorTwist", &rocalColorTwist,
          py::return_value_policy::reference);
    m.def("colorTwistFixed", &rocalColorTwistFixed,
          py::return_value_policy::reference);
    m.def("cropMirrorNormalize", &rocalCropMirrorNormalize,
          py::return_value_policy::reference);
    m.def("crop", &rocalCrop,
          py::return_value_policy::reference);
    m.def("cropFixed", &rocalCropFixed,
          py::return_value_policy::reference);
    m.def("centerCropFixed", &rocalCropCenterFixed,
          py::return_value_policy::reference);
    m.def("brightness", &rocalBrightness,
          py::return_value_policy::reference);
    m.def("brightnessFixed", &rocalBrightnessFixed,
          py::return_value_policy::reference);
    m.def("gammaCorrection", &rocalGamma,
          py::return_value_policy::reference);
    m.def("rain", &rocalRain,
          py::return_value_policy::reference);
    m.def("snow", &rocalSnow,
          py::return_value_policy::reference);
    m.def("blur", &rocalBlur,
          py::return_value_policy::reference);
    m.def("contrast", &rocalContrast,
          py::return_value_policy::reference);
    m.def("flip", &rocalFlip,
          py::return_value_policy::reference);
    m.def("jitter", &rocalJitter,
          py::return_value_policy::reference);
    m.def("rotate", &rocalRotate,
          py::return_value_policy::reference);
    m.def("hue", &rocalHue,
          py::return_value_policy::reference);
    m.def("saturation", &rocalSaturation,
          py::return_value_policy::reference);
    m.def("warpAffineFixed", &rocalWarpAffineFixed,
          py::return_value_policy::reference);
    m.def("fog", &rocalFog,
          py::return_value_policy::reference);
    m.def("fishEye", &rocalFishEye,
          py::return_value_policy::reference);
    m.def("vignette", &rocalVignette,
          py::return_value_policy::reference);
    m.def("snpNoise", &rocalSnPNoise,
          py::return_value_policy::reference);
    m.def("exposure", &rocalExposure,
          py::return_value_policy::reference);
    m.def("pixelate", &rocalPixelate,
          py::return_value_policy::reference);
    m.def("blend", &rocalBlend,
          py::return_value_policy::reference);
    m.def("randomCrop", &rocalRandomCrop,
          py::return_value_policy::reference);
    m.def("colorTemp", &rocalColorTemp,
          py::return_value_policy::reference);
    m.def("lensCorrection", &rocalLensCorrection,
          py::return_value_policy::reference);
}
}  // namespace rocal
