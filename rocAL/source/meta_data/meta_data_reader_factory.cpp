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

#include "meta_data_reader_factory.h"

#include <memory>

#include "caffe2_meta_data_reader.h"
#include "caffe2_meta_data_reader_detection.h"
#include "caffe_meta_data_reader.h"
#include "caffe_meta_data_reader_detection.h"
#include "cifar10_meta_data_reader.h"
#include "coco_meta_data_reader.h"
#include "coco_meta_data_reader_key_points.h"
#include "exception.h"
#include "label_reader_folders.h"
#include "mxnet_meta_data_reader.h"
#include "text_file_meta_data_reader.h"
#include "tf_meta_data_reader.h"
#include "tf_meta_data_reader_detection.h"
#include "video_label_reader.h"

std::shared_ptr<MetaDataReader> create_meta_data_reader(const MetaDataConfig& config, pMetaDataBatch& meta_data_batch) {
    switch (config.reader_type()) {
        case MetaDataReaderType::FOLDER_BASED_LABEL_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("FOLDER_BASED_LABEL_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<LabelReaderFolders>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
#ifdef ROCAL_VIDEO
        case MetaDataReaderType::VIDEO_LABEL_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("VIDEO_LABEL_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<VideoLabelReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
#endif
        case MetaDataReaderType::TEXT_FILE_META_DATA_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("TEXT_FILE_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<TextFileMetaDataReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::TF_META_DATA_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("TF_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<TFMetaDataReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::TF_DETECTION_META_DATA_READER: {
            if (config.type() != MetaDataType::BoundingBox)
                THROW("TF_DETECTION_META_DATA_READER can only be used to load bounding boxes")
            auto meta_data_reader = std::make_shared<TFMetaDataReaderDetection>();
            meta_data_batch = std::make_shared<BoundingBoxBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::COCO_META_DATA_READER: {
            if (config.type() != MetaDataType::BoundingBox && config.type() != MetaDataType::PolygonMask)
                THROW("COCO_META_DATA_READER can only be used to load bounding boxes and mask coordinates")
            auto meta_data_reader = std::make_shared<COCOMetaDataReader>();
            if (config.type() == MetaDataType::PolygonMask)
                meta_data_batch = std::make_shared<PolygonMaskBatch>();
            else
                meta_data_batch = std::make_shared<BoundingBoxBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::COCO_KEY_POINTS_META_DATA_READER: {
            if (config.type() != MetaDataType::KeyPoints)
                THROW("COCO_KEY_POINTS_META_DATA_READER can only be used to load keypoints")
            auto meta_data_reader = std::make_shared<COCOMetaDataReaderKeyPoints>();
            meta_data_batch = std::make_shared<KeyPointBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::CIFAR10_META_DATA_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("TEXT_FILE_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<Cifar10MetaDataReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::CAFFE_META_DATA_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("CAFFE_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<CaffeMetaDataReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::CAFFE_DETECTION_META_DATA_READER: {
            if (config.type() != MetaDataType::BoundingBox)
                THROW("CAFFE_DETECTION_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<CaffeMetaDataReaderDetection>();
            meta_data_batch = std::make_shared<BoundingBoxBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::CAFFE2_META_DATA_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("CAFFE2_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<Caffe2MetaDataReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::CAFFE2_DETECTION_META_DATA_READER: {
            if (config.type() != MetaDataType::BoundingBox)
                THROW("CAFFE2_DETECTION_META_DATA_READER can only be used to load labels")
            auto meta_data_reader = std::make_shared<Caffe2MetaDataReaderDetection>();
            meta_data_batch = std::make_shared<BoundingBoxBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        case MetaDataReaderType::MXNET_META_DATA_READER: {
            if (config.type() != MetaDataType::Label)
                THROW("MXNetMetaDataReader can only be used to load labels")
            auto meta_data_reader = std::make_shared<MXNetMetaDataReader>();
            meta_data_batch = std::make_shared<LabelBatch>();
            meta_data_reader->init(config, meta_data_batch);
            return meta_data_reader;
        } break;
        default:
            THROW("MetaDataReader type is unsupported : " + TOSTR(config.reader_type()));
    }
}
