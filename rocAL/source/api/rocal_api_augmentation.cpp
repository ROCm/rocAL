/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vx_ext_rpp_version.h>
#include "meta_data/augmentations_meta_nodes.h"
#include "augmentations/augmentations_nodes.h"
#include "pipeline/commons.h"
#include "pipeline/context.h"
#include "loaders/image_source_evaluator.h"
#include "rocal_api.h"

RocalTensor ROCAL_API_CALL
rocalSequenceRearrange(RocalContext p_context,
                       RocalTensor p_input,
                       std::vector<unsigned int>& new_order,
                       bool is_output) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto input = static_cast<Tensor*>(p_input);
    auto context = static_cast<Context*>(p_context);
    try {
        if (new_order.size() == 0)
            THROW("The new order for the sequence passed should be greater than 0")
        TensorInfo output_info = input->info();
        std::vector<size_t> new_dims;
        new_dims = output_info.dims();
        new_dims[1] = new_order.size();
        output_info.set_dims(new_dims);

        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<SequenceRearrangeNode> sequence_rearrange_node = context->master_graph->add_node<SequenceRearrangeNode>({input}, {output});
        sequence_rearrange_node->init(new_order);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRotate(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_angle,
    unsigned dest_width,
    unsigned dest_height,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto angle = static_cast<FloatParam*>(p_angle);
    try {
        if (dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the rotate node, user can create a tensor with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<RotateNode> rotate_node = context->master_graph->add_node<RotateNode>({input}, {output});
        rotate_node->init(angle, static_cast<ResizeInterpolationType>(interpolation_type));
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<RotateMetaNode, RotateNode>(rotate_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRotateFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float angle,
    bool is_output,
    unsigned dest_width,
    unsigned dest_height,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the rotate node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<RotateNode> rotate_node = context->master_graph->add_node<RotateNode>({input}, {output});
        rotate_node->init(angle, static_cast<ResizeInterpolationType>(interpolation_type));
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<RotateMetaNode, RotateNode>(rotate_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGamma(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_gamma,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto gamma = static_cast<FloatParam*>(p_gamma);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GammaNode>({input}, {output})->init(gamma);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGammaFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float gamma,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GammaNode>({input}, {output})->init(gamma);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalHue(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_hue,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto hue = static_cast<FloatParam*>(p_hue);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalHueFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float hue,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSaturation(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_saturation,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto saturation = static_cast<FloatParam*>(p_saturation);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SaturationNode>({input}, {output})->init(saturation);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSaturationFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float saturation,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SaturationNode>({input}, {output})->init(saturation);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropResize(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned dest_width, unsigned dest_height,
    bool is_output,
    RocalFloatParam p_area,
    RocalFloatParam p_aspect_ratio,
    RocalFloatParam p_x_center_drift,
    RocalFloatParam p_y_center_drift,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto area = static_cast<FloatParam*>(p_area);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_center_drift = static_cast<FloatParam*>(p_x_center_drift);
    auto y_center_drift = static_cast<FloatParam*>(p_y_center_drift);
    try {
        if (dest_width == 0 || dest_height == 0)
            THROW("CropResize node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the crop resize node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);

        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropResizeNode> crop_resize_node = context->master_graph->add_node<CropResizeNode>({input}, {output});
        crop_resize_node->init(area, aspect_ratio, x_center_drift, y_center_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode, CropResizeNode>(crop_resize_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRandomResizedCrop(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned dest_width, unsigned dest_height,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0)
            THROW("rocalRandomResizedCrop node needs to receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the crop resize node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);

        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropResizeNode> random_resize_crop_node = context->master_graph->add_node<CropResizeNode>({input}, {output});
        random_resize_crop_node->init(area_factor, aspect_ratio, static_cast<ResizeInterpolationType>(interpolation_type));
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode, CropResizeNode>(random_resize_crop_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropResizeFixed(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned dest_width, unsigned dest_height,
    bool is_output,
    float area,
    float aspect_ratio,
    float x_center_drift,
    float y_center_drift,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0)
            THROW("CropResize node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the crop resize node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropResizeNode> crop_resize_node = context->master_graph->add_node<CropResizeNode>({input}, {output});
        crop_resize_node->init(area, aspect_ratio, x_center_drift, y_center_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode, CropResizeNode>(crop_resize_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalROIResize(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned dest_width,
    unsigned dest_height,
    bool is_output,
    unsigned roi_h,
    unsigned roi_w,
    float roi_pos_x,
    float roi_pos_y,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0)
            THROW("ROI Resize needs to receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the ROI resize augmentation, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropResizeNode> roi_resize_node = context->master_graph->add_node<CropResizeNode>({input}, {output});
        roi_resize_node->init(roi_h, roi_w, roi_pos_x, roi_pos_y, static_cast<ResizeInterpolationType>(interpolation_type));
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode, CropResizeNode>(roi_resize_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalResize(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned dest_width,
    unsigned dest_height,
    bool is_output,
    RocalResizeScalingMode scaling_mode,
    std::vector<unsigned> max_size,
    unsigned resize_shorter,
    unsigned resize_longer,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if ((dest_width | dest_height | resize_longer | resize_shorter) == 0)
            THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
        if ((dest_width | dest_height) && (resize_longer | resize_shorter))
            THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
        if (resize_longer && resize_shorter)
            THROW("'resize_longer' and 'resize_shorter' cannot be passed together. They are mutually exclusive.")

        unsigned out_width, out_height;
        RocalResizeScalingMode resize_scaling_mode;

        // Change the scaling mode if resize_shorter or resize_longer is specified
        if (resize_shorter) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
            out_width = out_height = resize_shorter;
        } else if (resize_longer) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
            out_width = out_height = resize_longer;
        } else {
            resize_scaling_mode = scaling_mode;
            out_width = dest_width;
            out_height = dest_height;
        }

        std::vector<unsigned> maximum_size;
        if (max_size.size()) {
            if (max_size.size() == 1) {
                maximum_size = {max_size[0], max_size[0]};
            } else if (max_size.size() == 2) {
                maximum_size = {max_size[0], max_size[1]};  // {width, height}
            } else {
                THROW("The length of max_size vector exceeds the image dimension.")
            }
        }

        // Determine the max width and height to be set to the output info
        unsigned max_out_width, max_out_height;
        if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
            // If max_size is passed by the user, the resized images cannot exceed the max size,
            max_out_width = maximum_size[0];
            max_out_height = maximum_size[1];
        } else {
            // compute the output info width and height wrt the scaling modes and roi passed
            if (resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                max_out_width = out_width ? out_width : input->info().max_shape()[0];
                max_out_height = out_height ? out_height : input->info().max_shape()[1];
            } else if (resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
            } else {
                max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
            }
            if (maximum_size.size() == 2) {
                max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
            }
        }

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);
        output_info.modify_dims_width_and_height(op_tensor_layout, max_out_width, max_out_height);

        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<ResizeNode> resize_node = context->master_graph->add_node<ResizeNode>({input}, {output});
        resize_node->init(out_width, out_height, static_cast<ResizeScalingMode>(resize_scaling_mode), maximum_size, static_cast<ResizeInterpolationType>(interpolation_type));
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMetaNode, ResizeNode>(resize_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
    ROCAL_API_CALL
    rocalResizeMirrorNormalize(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned dest_width,
        unsigned dest_height,
        std::vector<float>& mean,
        std::vector<float>& std_dev,
        bool is_output,
        RocalResizeScalingMode scaling_mode,
        std::vector<unsigned> max_size,
        unsigned resize_shorter,
        unsigned resize_longer,
        RocalResizeInterpolationType interpolation_type,
        RocalIntParam p_mirror,
        RocalTensorLayout output_layout,
        RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto mirror = static_cast<IntParam*>(p_mirror);

    try {
        if ((dest_width | dest_height | resize_longer | resize_shorter) == 0)
            THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
        // MaskRCNN training uses a new resize scaling mode - MIN_MAX_SCALING_MODE where min_size and max_size is passed and the final output size is calculated from the image size
        // Only in the case of MIN_MAX_SCALING_MODE, both resize_shorter and resize_longer values can be passed together
        if ((dest_width | dest_height) && (resize_longer | resize_shorter) && (scaling_mode != RocalResizeScalingMode::ROCAL_SCALING_MODE_MIN_MAX))
            THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
        if (resize_longer && resize_shorter && scaling_mode != RocalResizeScalingMode::ROCAL_SCALING_MODE_MIN_MAX)
            THROW("'resize_longer' and 'resize_shorter' can only be passed together for min max scaling mode")

        unsigned out_width, out_height;
        RocalResizeScalingMode resize_scaling_mode;

        // Change the scaling mode if resize_shorter or resize_longer is specified
        if (scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_MIN_MAX) {
            resize_scaling_mode = scaling_mode;
            out_width = dest_width;
            out_height = dest_height;
        } else if (resize_shorter) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
            out_width = out_height = resize_shorter;
        } else if (resize_longer) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
            out_width = out_height = resize_longer;
        } else {
            resize_scaling_mode = scaling_mode;
            out_width = dest_width;
            out_height = dest_height;
        }

        std::vector<unsigned> maximum_size;
        if (max_size.size()) {
            if (max_size.size() == 1) {
                maximum_size = {max_size[0], max_size[0]};
            } else if (max_size.size() == 2) {
                maximum_size = {max_size[0], max_size[1]};  // {width, height}
            } else {
                THROW("The length of max_size vector exceeds the image dimension.")
            }
        }

        // Determine the max width and height to be set to the output info
        unsigned max_out_width, max_out_height;
        if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
            // If max_size is passed by the user, the resized images cannot exceed the max size,
            max_out_width = maximum_size[0];
            max_out_height = maximum_size[1];
        } else {
            // compute the output info width and height wrt the scaling modes and roi passed
            if (resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                max_out_width = out_width ? out_width : input->info().max_shape()[0];
                max_out_height = out_height ? out_height : input->info().max_shape()[1];
            } else if (resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
            } else {
                max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
            }
            if (maximum_size.size() == 2) {
                max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
            }
        }
        if (scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_MIN_MAX) {
            // For Min Max scaling mode, both min size and max size are passed as resize_shorter and resize_longer values
            maximum_size = {resize_shorter, resize_longer};
        }

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);
        output_info.modify_dims_width_and_height(op_tensor_layout, max_out_width, max_out_height);

        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<ResizeMirrorNormalizeNode> rmn_node = context->master_graph->add_node<ResizeMirrorNormalizeNode>({input}, {output});
        rmn_node->init(out_width, out_height, static_cast<ResizeScalingMode>(resize_scaling_mode), maximum_size, static_cast<ResizeInterpolationType>(interpolation_type), mean, std_dev, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMirrorNormalizeMetaNode, ResizeMirrorNormalizeNode>(rmn_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBrightness(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_alpha,
    RocalFloatParam p_beta,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBrightnessFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float alpha,
    float beta,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlur(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlurNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlend(
    RocalContext p_context,
    RocalTensor p_input1,
    RocalTensor p_input2,
    bool is_output,
    RocalFloatParam p_ratio,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);

    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    auto ratio = static_cast<FloatParam*>(p_ratio);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlendNode>({input1, input2}, {output})->init(ratio);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlendFixed(
    RocalContext p_context,
    RocalTensor p_input1,
    RocalTensor p_input2,
    float ratio,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);

    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlendNode>({input1, input2}, {output})->init(ratio);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalWarpAffine(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    unsigned dest_height, unsigned dest_width,
    RocalFloatParam p_x0, RocalFloatParam p_x1,
    RocalFloatParam p_y0, RocalFloatParam p_y1,
    RocalFloatParam p_o0, RocalFloatParam p_o1,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto x0 = static_cast<FloatParam*>(p_x0);
    auto x1 = static_cast<FloatParam*>(p_x1);
    auto y0 = static_cast<FloatParam*>(p_y0);
    auto y1 = static_cast<FloatParam*>(p_y1);
    auto o0 = static_cast<FloatParam*>(p_o0);
    auto o1 = static_cast<FloatParam*>(p_o1);
    try {
        if (dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the warp affine node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(x0, x1, y0, y1, o0, o1, static_cast<ResizeInterpolationType>(interpolation_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalWarpAffineFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float x0, float x1,
    float y0, float y1,
    float o0, float o1,
    bool is_output,
    unsigned int dest_height,
    unsigned int dest_width,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the warp affine node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(x0, x1, y0, y1, o0, o1, static_cast<ResizeInterpolationType>(interpolation_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFishEye(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FisheyeNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalVignette(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_sdev,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto sdev = static_cast<FloatParam*>(p_sdev);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalVignetteFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float sdev,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJitter(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalIntParam p_kernel_size,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto kernel_size = static_cast<IntParam*>(p_kernel_size);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<JitterNode>({input}, {output})->init(kernel_size, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJitterFixed(
    RocalContext p_context,
    RocalTensor p_input,
    int kernel_size,
    bool is_output,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<JitterNode>({input}, {output})->init(kernel_size, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnPNoise(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_noise_prob,
    RocalFloatParam p_salt_prob,
    RocalFloatParam p_salt_val,
    RocalFloatParam p_pepper_val,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto noise_probability = static_cast<FloatParam*>(p_noise_prob);
    auto salt_probability = static_cast<FloatParam*>(p_salt_prob);
    auto salt_value = static_cast<FloatParam*>(p_salt_val);
    auto pepper_value = static_cast<FloatParam*>(p_pepper_val);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnPNoiseNode>({input}, {output})->init(noise_probability, salt_probability, salt_value, pepper_value, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnPNoiseFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float noise_prob,
    float salt_prob,
    float salt_val,
    float pepper_val,
    bool is_output,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnPNoiseNode>({input}, {output})->init(noise_prob, salt_prob, salt_val, pepper_val, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGaussianNoise(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_mean,
    RocalFloatParam p_stddev,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalGaussianNoise requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto mean = static_cast<FloatParam*>(p_mean);
    auto stddev = static_cast<FloatParam*>(p_stddev);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GaussianNoiseNode>({input}, {output})->init(mean, stddev, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGaussianNoiseFixed(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float mean,
    float stddev,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalGaussianNoiseFixed requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GaussianNoiseNode>({input}, {output})->init(mean, stddev, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalShotNoise(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_noise_factor,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalShotNoise requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto noise_factor = static_cast<FloatParam*>(p_noise_factor);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ShotNoiseNode>({input}, {output})->init(noise_factor, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalShotNoiseFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float noise_factor,
    bool is_output,
    int seed,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalShotNoiseFixed requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ShotNoiseNode>({input}, {output})->init(noise_factor, seed);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalLUT(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalLUT requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<LutNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPosterize(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalIntParam p_num_bits,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalPosterize requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto num_bits = static_cast<IntParam*>(p_num_bits);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<PosterizeNode>({input}, {output})->init(num_bits);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPosterizeFixed(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned int num_bits,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalPosterizeFixed requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<PosterizeNode>({input}, {output})->init(num_bits);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSolarize(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_threshold,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalSolarize requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto threshold = static_cast<FloatParam*>(p_threshold);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SolarizeNode>({input}, {output})->init(threshold);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSolarizeFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float threshold,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalSolarizeFixed requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SolarizeNode>({input}, {output})->init(threshold);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCompressionDistortion(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalIntParam p_quality,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalJpegCompressionDistortion requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto quality = static_cast<IntParam*>(p_quality);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<JpegCompressionDistortionNode>({input}, {output})->init(quality);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCompressionDistortionFixed(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned int quality,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalJpegCompressionDistortionFixed requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<JpegCompressionDistortionNode>({input}, {output})->init(quality);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFlip(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalIntParam p_horizontal_flag,
    RocalIntParam p_vertical_flag,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto horizontal_flag = static_cast<IntParam*>(p_horizontal_flag);
    auto vertical_flag = static_cast<IntParam*>(p_vertical_flag);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<FlipNode> flip_node = context->master_graph->add_node<FlipNode>({input}, {output});
        flip_node->init(horizontal_flag, vertical_flag);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<FlipMetaNode, FlipNode>(flip_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFlipFixed(
    RocalContext p_context,
    RocalTensor p_input,
    int horizontal_flag,
    int vertical_flag,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<FlipNode> flip_node = context->master_graph->add_node<FlipNode>({input}, {output});
        flip_node->init(horizontal_flag, vertical_flag);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<FlipMetaNode, FlipNode>(flip_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalContrast(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_contrast_factor,
    RocalFloatParam p_contrast_center,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto contrast_factor = static_cast<FloatParam*>(p_contrast_factor);
    auto contrast_center = static_cast<FloatParam*>(p_contrast_center);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ContrastNode>({input}, {output})->init(contrast_factor, contrast_center);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalContrastFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float contrast_factor,
    float contrast_center,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ContrastNode>({input}, {output})->init(contrast_factor, contrast_center);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnow(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_snow_value,
    RocalFloatParam p_brightness_coefficient,
    RocalIntParam p_dark_mode,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto snow_value = static_cast<FloatParam*>(p_snow_value);
    auto brightness_coefficient = static_cast<FloatParam*>(p_brightness_coefficient);
    auto dark_mode = static_cast<IntParam*>(p_dark_mode);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnowNode>({input}, {output})->init(snow_value, brightness_coefficient, dark_mode);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnowFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float snow_value,
    bool is_output,
    float brightness_coefficient,
    int dark_mode,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnowNode>({input}, {output})->init(snow_value, brightness_coefficient, dark_mode);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRain(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float rain_percentage,
    int rain_width,
    int rain_height,
    float rain_slant_angle,
    RocalFloatParam p_rain_transparency,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto rain_transparency = static_cast<FloatParam*>(p_rain_transparency);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_percentage, rain_width, rain_height, rain_slant_angle, rain_transparency);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRainFixed(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float rain_percentage,
    int rain_width,
    int rain_height,
    float rain_slant_angle,
    float rain_transparency,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_percentage, rain_width, rain_height, rain_slant_angle, rain_transparency);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSpatter(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalIntParam p_red,
    RocalIntParam p_green,
    RocalIntParam p_blue,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalSpatter requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto red = static_cast<IntParam*>(p_red);
    auto green = static_cast<IntParam*>(p_green);
    auto blue = static_cast<IntParam*>(p_blue);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SpatterNode>({input}, {output})->init(red, green, blue);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSpatterFixed(
    RocalContext p_context,
    RocalTensor p_input,
    uint8_t red,
    uint8_t green,
    uint8_t blue,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalSpatterFixed requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SpatterNode>({input}, {output})->init(red, green, blue);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTemp(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalIntParam p_adj_value_param,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto adj_value_param = static_cast<IntParam*>(p_adj_value_param);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adj_value_param);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTempFixed(
    RocalContext p_context,
    RocalTensor p_input,
    int adj_value_param,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adj_value_param);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorJitter(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_brightness,
    RocalFloatParam p_contrast,
    RocalFloatParam p_hue,
    RocalFloatParam p_saturation,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalColorJitter requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto brightness = static_cast<FloatParam*>(p_brightness);
    auto contrast = static_cast<FloatParam*>(p_contrast);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto saturation = static_cast<FloatParam*>(p_saturation);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorJitterNode>({input}, {output})->init(brightness, contrast, hue, saturation);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorJitterFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float brightness,
    float contrast,
    float hue,
    float saturation,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalColorJitterFixed requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorJitterNode>({input}, {output})->init(brightness, contrast, hue, saturation);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalChannelPermute(
    RocalContext p_context,
    RocalTensor p_input,
    std::vector<unsigned int>& permutation,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalChannelPermute requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ChannelPermuteNode>({input}, {output})->init(permutation);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFog(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_intensity_param,
    RocalFloatParam p_gray_param,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto intensity_param = static_cast<FloatParam*>(p_intensity_param);
    auto gray_param = static_cast<FloatParam*>(p_gray_param);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FogNode>({input}, {output})->init(intensity_param, gray_param);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFogFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float intensity_param,
    float gray_param,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FogNode>({input}, {output})->init(intensity_param, gray_param);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPixelate(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float pixelate_percent,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<PixelateNode>({input}, {output})->init(pixelate_percent);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalLensCorrection(
    RocalContext p_context,
    RocalTensor p_input,
    std::vector<CameraMatrix> camera_matrix,
    std::vector<DistortionCoeffs> distortion_coeffs,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        if (camera_matrix.size() != context->user_batch_size() && camera_matrix.size() != 1) {
            THROW("Camera matrix size must be either 1 (for all images) or equal to batch size (" + std::to_string(context->user_batch_size()) + "). Provided size: " + std::to_string(camera_matrix.size()))
        }
        if (distortion_coeffs.size() != context->user_batch_size() && distortion_coeffs.size() != 1) {
            THROW("Distortion coefficients size must be either 1 (for all images) or equal to batch size (" + std::to_string(context->user_batch_size()) + "). Provided size: " + std::to_string(distortion_coeffs.size()))
        }
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<LensCorrectionNode>({input}, {output})->init(camera_matrix, distortion_coeffs);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalExposure(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_exposure_factor,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto exposure_factor = static_cast<FloatParam*>(p_exposure_factor);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ExposureNode>({input}, {output})->init(exposure_factor);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalExposureFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float exposure_factor,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ExposureNode>({input}, {output})->init(exposure_factor);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTwist(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_alpha,
    RocalFloatParam p_beta,
    RocalFloatParam p_hue,
    RocalFloatParam p_sat,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue, sat);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTwistFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float alpha,
    float beta,
    float hue,
    float sat,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue, sat);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, unsigned crop_height,
                         unsigned crop_width, float start_x, float start_y, std::vector<float>& mean,
                         std::vector<float>& std_dev, bool is_output, RocalIntParam p_mirror,
                         RocalTensorLayout output_layout,
                         RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto mirror = static_cast<IntParam*>(p_mirror);
    try {
        if (crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the crop mirror normalize resize node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, crop_width, crop_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<CropMirrorNormalizeNode> cmn_node = context->master_graph->add_node<CropMirrorNormalizeNode>({input}, {output});
        cmn_node->init(crop_height, crop_width, start_x, start_y, mean, std_dev, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMirrorNormalizeMetaNode, CropMirrorNormalizeNode>(cmn_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCrop(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_crop_width,
    RocalFloatParam p_crop_height,
    RocalFloatParam p_crop_depth,
    RocalFloatParam p_crop_pox_x,
    RocalFloatParam p_crop_pos_y,
    RocalFloatParam p_crop_pos_z,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_h, crop_w, x_drift, y_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode, CropNode>(crop_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropFixed(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned crop_width,
    unsigned crop_height,
    unsigned crop_depth,
    bool is_output,
    float crop_pos_x,
    float crop_pos_y,
    float crop_pos_z,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs to receive non-zero destination dimensions")
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the crop node, user can create an tensor with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, crop_width, crop_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width, crop_pos_x, crop_pos_y);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode, CropNode>(crop_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropCenterFixed(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned crop_width,
    unsigned crop_height,
    unsigned crop_depth,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs to receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the crop node, user can create an tensor with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, crop_width, crop_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode, CropNode>(crop_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalResizeCropMirrorFixed(
    RocalContext p_context,
    RocalTensor p_input,
    unsigned dest_width,
    unsigned dest_height,
    bool is_output,
    unsigned crop_h,
    unsigned crop_w,
    RocalIntParam p_mirror,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto mirror = static_cast<IntParam*>(p_mirror);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0)
            THROW("Crop Mirror node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the resize_crop_mirror node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<ResizeCropMirrorNode> rcm_node = context->master_graph->add_node<ResizeCropMirrorNode>({input}, {output});
        rcm_node->init(crop_h, crop_w, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeCropMirrorMetaNode, ResizeCropMirrorNode>(rcm_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL rocalResizeCropMirror(
    RocalContext p_context, RocalTensor p_input,
    unsigned dest_width, unsigned dest_height,
    bool is_output, RocalFloatParam p_crop_height,
    RocalFloatParam p_crop_width, RocalIntParam p_mirror,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto mirror = static_cast<IntParam*>(p_mirror);
    try {
        if (dest_width == 0 || dest_height == 0)
            THROW("Crop Mirror node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);

        // For the resize_crop_mirror node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<ResizeCropMirrorNode> rcm_node = context->master_graph->add_node<ResizeCropMirrorNode>({input}, {output});
        rcm_node->init(crop_h, crop_w, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeCropMirrorMetaNode, ResizeCropMirrorNode>(rcm_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRandomCrop(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_crop_area_factor,
    RocalFloatParam p_crop_aspect_ratio,
    RocalFloatParam p_crop_pox_x,
    RocalFloatParam p_crop_pos_y,
    int num_of_attempts,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_area_factor = static_cast<FloatParam*>(p_crop_area_factor);
    auto crop_aspect_ratio = static_cast<FloatParam*>(p_crop_aspect_ratio);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<RandomCropNode> crop_node = context->master_graph->add_node<RandomCropNode>({input}, {output});
        crop_node->init(crop_area_factor, crop_aspect_ratio, x_drift, y_drift, num_of_attempts);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<SSDRandomCropMetaNode,RandomCropNode>(crop_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSSDRandomCrop(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_threshold,
    RocalFloatParam p_crop_area_factor,
    RocalFloatParam p_crop_aspect_ratio,
    RocalFloatParam p_crop_pox_x,
    RocalFloatParam p_crop_pos_y,
    int num_of_attempts,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_area_factor = static_cast<FloatParam*>(p_crop_area_factor);
    auto crop_aspect_ratio = static_cast<FloatParam*>(p_crop_aspect_ratio);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);

        std::shared_ptr<SSDRandomCropNode> crop_node = context->master_graph->add_node<SSDRandomCropNode>({input}, {output});
        crop_node->init(crop_area_factor, crop_aspect_ratio, x_drift, y_drift, num_of_attempts);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<SSDRandomCropMetaNode,SSDRandomCropNode>(crop_node);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCopy(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNop(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<NopNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPreEmphasisFilter(RocalContext p_context,
                       RocalTensor p_input,
                       bool is_output,
                       RocalFloatParam p_preemph_coeff,
                       RocalAudioBorderType preemph_border_type,
                       RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto preemph_coeff = static_cast<FloatParam*>(p_preemph_coeff);
    try {
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        if (op_tensor_datatype != RocalTensorDataType::FP32) {
            THROW("Only FP32 dtype is supported for PreEmphasis filter augmentation.")
        }
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<PreemphasisFilterNode>({input}, {output})->init(preemph_coeff, static_cast<AudioBorderType>(preemph_border_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSpectrogram(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        std::vector<float> &window_fn,
        bool center_windows,
        bool reflect_padding,
        int power,
        int nfft,
        int window_length,
        int window_step,
        RocalTensorLayout output_layout,
        RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(output_datatype);
        if (op_tensor_data_type != RocalTensorDataType::FP32) {
            WRN("Only FP32 data-type is supported for Spectrogram augmentation.")
            op_tensor_data_type = RocalTensorDataType::FP32;
        }
        std::vector<size_t> max_dims = input->info().max_shape();
        if (max_dims[1] != 1) THROW("Spectrogram only supports single channel inputs. Please check the input passed.")
        int window_offset = (!center_windows) ? window_length :  0;
        int max_frame = (((max_dims[0] - window_offset) / window_step) + 1);
        max_frame = std::max(0, max_frame);
        int bins = std::max(0, (nfft / 2) + 1);
        std::vector<size_t> dims = input->info().dims();
        RocalTensorlayout spectrogram_layout = static_cast<RocalTensorlayout>(output_layout);
        if (spectrogram_layout == RocalTensorlayout::NTF) {
            dims[1] = max_frame;
            dims[2] = bins;
        } else if (spectrogram_layout == RocalTensorlayout::NFT) {
            dims[1] = bins;
            dims[2] = max_frame;
        } else {
            THROW("Spectrogram supports only NFT / NTF layouts")
        }
        TensorInfo output_info = TensorInfo(std::vector<size_t>(std::move(dims)),
                                            context->master_graph->mem_type(),
                                            op_tensor_data_type,
                                            spectrogram_layout);
        if(power != 1 && power != 2) {
            WRN("rocalSpectrogram power value can be 1 or 2, setting it to default 2")
            power = 2;
        }
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SpectrogramNode>({input}, {output})->init(center_windows, reflect_padding,
                                                                                  power, nfft, window_length,
                                                                                  window_step, window_fn);
    } catch(const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalToDecibels(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float cutoff_db,
    float multiplier,
    float reference_magnitude,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        if (op_tensor_data_type != RocalTensorDataType::FP32) {
            THROW("Only FP32 dtype is supported for To decibels augmentation.")
        }
        output_info.set_data_type(op_tensor_data_type);
        if (input->info().layout() == RocalTensorlayout::NFT || input->info().layout() == RocalTensorlayout::NTF) // Layout is changed when input is from spectrogram/mel filter bank
            output_info.set_tensor_layout(RocalTensorlayout::NHW);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ToDecibelsNode>({input}, {output})->init(cutoff_db, multiplier, reference_magnitude);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalResample(RocalContext p_context,
              RocalTensor p_input,
              RocalTensor p_output_resample_rate,
              bool is_output,
              float sample_hint,
              float quality,
              RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr) || (p_output_resample_rate == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    Tensor* resampled_output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto output_resample_rate = static_cast<Tensor*>(p_output_resample_rate);
    try {
        TensorInfo output_info = input->info();
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(output_datatype);
        if (op_tensor_data_type != RocalTensorDataType::FP32) {
            THROW("Only FP32 dtype is supported for resample augmentation.")
        }
        output_info.set_data_type(op_tensor_data_type);
        if (sample_hint > 0) {
            std::vector<size_t> max_dims = output_info.max_shape();
            std::vector<size_t> dims = output_info.dims();
            dims[1] = std::ceil(sample_hint);
            dims[2] = max_dims[1];
            output_info.set_dims(dims);
        } else {
            THROW("Please pass a valid resample hint")
        }
        resampled_output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ResampleNode>({input}, {resampled_output})->init(output_resample_rate, quality);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return resampled_output;
}

RocalTensor rocalTensorMulScalar(RocalContext p_context,
                                 RocalTensor p_input,
                                 bool is_output,
                                 float scalar,
                                 RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(output_datatype);
        if (op_tensor_data_type != RocalTensorDataType::FP32) {
            THROW("Only FP32 dtype is supported for TensorMulScalar augmentation.")
        }
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_data_type);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<TensorMulScalarNode>({input}, {output})->init(scalar);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor rocalTensorAddTensor(RocalContext p_context,
                                 RocalTensor p_input1,
                                 RocalTensor p_input2,
                                 bool is_output,
                                 RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);
    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(output_datatype);
        if (op_tensor_data_type != RocalTensorDataType::FP32) {
            THROW("Only FP32 dtype is supported for TensorAddTensor augmentation.")
        }
        TensorInfo output_info = input1->info();
        output_info.set_data_type(op_tensor_data_type);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<TensorAddTensorNode>({input1, input2}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor rocalUniformDistribution(RocalContext p_context,
                                     RocalTensor p_input,
                                     bool is_output,
                                     std::vector<float>& range) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        std::vector<size_t> dims = {context->user_batch_size(), 1};
        auto info = TensorInfo(dims,
                               context->master_graph->mem_type(),
                               RocalTensorDataType::FP32);
        info.set_dims(dims);
        output = context->master_graph->create_internal_tensor(info);
        context->master_graph->add_node<UniformDistributionNode>({input}, {output})->init(range);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor rocalNormalDistribution(RocalContext p_context,
                                    RocalTensor p_input,
                                    bool is_output,
                                    float mean,
                                    float stddev) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        std::vector<size_t> dims = {context->user_batch_size(), 1};
        auto info = TensorInfo(dims,
                               context->master_graph->mem_type(),
                               RocalTensorDataType::FP32);
        info.set_dims(dims);
        output = context->master_graph->create_internal_tensor(info);
        context->master_graph->add_node<NormalDistributionNode>({input}, {output})->init(mean, stddev);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalNSROutput ROCAL_API_CALL
rocalNonSilentRegionDetection(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float cutoff_db,
    float reference_power,
    int reset_interval,
    int window_length) {
    Tensor* anchor_output = nullptr;
    Tensor* shape_output = nullptr;
    RocalNSROutput output;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        std::vector<size_t> dims1 = {context->user_batch_size(), 1};
        auto info1 = TensorInfo(std::vector<size_t>(std::move(dims1)),
                                context->master_graph->mem_type(),
                                RocalTensorDataType::INT32);
        info1.set_max_shape();
        std::vector<size_t> dims2 = {context->user_batch_size(), 1};
        auto info2 = TensorInfo(std::vector<size_t>(std::move(dims2)),
                                context->master_graph->mem_type(),
                                RocalTensorDataType::INT32);
        info2.set_max_shape();
        anchor_output = context->master_graph->create_tensor(info1, is_output);
        shape_output = context->master_graph->create_tensor(info2, is_output);
        context->master_graph->add_node<NonSilentRegionDetectionNode>({input}, {anchor_output, shape_output})->init(cutoff_db, reference_power, window_length, reset_interval);
        output.anchor = anchor_output;
        output.shape = shape_output;
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }

    return output;
}

RocalTensor ROCAL_API_CALL
rocalSlice(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalTensor p_anchor,
    RocalTensor p_shape,
    std::vector<float> fill_values,
    RocalOutOfBoundsPolicy policy,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto anchor = static_cast<Tensor*>(p_anchor);
    auto shape = static_cast<Tensor*>(p_shape);
    try {
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_data_type);
        output_info.set_max_shape();
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SliceNode>({input}, {output})->init(anchor, shape, fill_values, static_cast<OutOfBoundsPolicy>(policy));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNormalize(RocalContext p_context, RocalTensor p_input, std::vector<unsigned>& axes,
               std::vector<float>& mean, std::vector<float>& std_dev, bool is_output,
               float scale, float shift,
               RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<NormalizeNode> normalize_node = context->master_graph->add_node<NormalizeNode>({input}, {output});
        normalize_node->init(axes, mean, std_dev, scale, shift);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalMelFilterBank(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    float freq_high,
    float freq_low,
    RocalMelScaleFormula mel_formula,
    int nfilter,
    bool normalize,
    float sample_rate,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_data_type = (RocalTensorDataType)output_datatype;
        if (op_tensor_data_type != RocalTensorDataType::FP32) {
            THROW("Only FP32 dtype is supported for MelFilterBank augmentation.")
        }
        TensorInfo output_info = input->info();
        std::vector<size_t> max_dims = output_info.max_shape();
        int max_frame = std::max(0ul, max_dims[1]);
        std::vector<size_t> dims = output_info.dims();
        dims[1] = nfilter;
        dims[2] = max_frame;
        output_info.set_dims(dims);
        output_info.set_data_type(op_tensor_data_type);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<MelFilterBankNode>({input}, {output})->init(freq_high, freq_low, static_cast<MelScaleFormula>(mel_formula), nfilter, normalize, sample_rate);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalTranspose(
    RocalContext p_context,
    RocalTensor p_input,
    std::vector<unsigned> perm,
    bool is_output,
    RocalTensorLayout output_layout) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        TensorInfo output_info = input->info();
        if (perm.size() != (output_info.num_of_dims() - 1)) {
            THROW("Transpose permutation must match with the input dims")
        }
        std::vector<size_t> dims = output_info.dims();
        for (int i = 1; i < dims.size(); i++)
            dims[i] = output_info.dims()[perm[i - 1] + 1];  // Perm contains permutation without batch dimension so adding +1
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_dims(dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<TransposeNode> transpose_node = context->master_graph->add_node<TransposeNode>({input}, {output});
        transpose_node->init(perm);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor rocalLog1p(RocalContext p_context,
                             RocalTensor p_input,
                             bool is_output) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_data_type = static_cast<RocalTensorDataType>(input->data_type());
        if (op_tensor_data_type != RocalTensorDataType::INT16) {
            THROW("Log1p augmentation only supported for int16 inputs")
        }
        op_tensor_data_type = RocalTensorDataType::FP32;  // Log1p only supports F32 outputs so setting output dtype to F32
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_data_type);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<Log1pNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPythonFunction(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        unsigned long long function_id,
        std::vector<size_t> output_dims,
        RocalTensorLayout output_layout,
        RocalTensorOutputType output_datatype)
{
#ifdef ROCAL_PYTHON_FUNCTION
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);

        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);
        output_info.set_tensor_layout(op_tensor_layout);
        // If user passes dimensions of the output tensor, set the TensorInfo dims accordingly
        // The user passed output dimensions won't contain the batch dimension, so it should be added
        if (!output_dims.empty()) {
            std::vector<size_t> dims = output_info.dims();
            if (output_dims.size() != dims.size() - 1)
                THROW("User passed output dimensions size does not match with the input tensor dimensions")
            for (size_t i = 1; i < dims.size(); i++)
                dims[i] = output_dims[i - 1];
            if (dims != output_info.dims())
                output_info.set_dims(dims);  // Only modify output tensor dims if it does not match the user specified dims
        }

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<PythonFunctionNode>({input}, {output})->init(function_id);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
#else
        THROW("PythonFunction node is not enabled since python/pybind11 is not present")
#endif
}

RocalTensor ROCAL_API_CALL
rocalColorCast(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_alpha,
    std::vector<float>& rgb,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorCastNode>({input}, {output})->init(alpha, rgb);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorCastFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float alpha,
    std::vector<float>& rgb,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorCastNode>({input}, {output})->init(alpha, rgb);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalMedianFilter(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    unsigned kernel_size,
    RocalImageBorderType border_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph
            ->add_node<MedianFilterNode>({input}, {output})
            ->init(kernel_size, static_cast<ImageBorderType>(border_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGaussianFilter(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_stddev,
    unsigned kernel_size,
    RocalImageBorderType border_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    auto stddev  = static_cast<FloatParam*>(p_stddev);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph
            ->add_node<GaussianFilterNode>({input}, {output})
            ->init(stddev, kernel_size, static_cast<ImageBorderType>(border_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGaussianFilterFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float stddev,
    unsigned kernel_size,
    RocalImageBorderType border_type,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph
            ->add_node<GaussianFilterNode>({input}, {output})
            ->init(stddev, kernel_size, static_cast<ImageBorderType>(border_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGridMask(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    unsigned tile_width,
    float grid_ratio,
    float grid_angle,
    unsigned translate_x,
    unsigned translate_y,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GridMaskNode>({input}, {output})->init(tile_width, grid_ratio, grid_angle, translate_x, translate_y);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNonLinearBlend(
    RocalContext p_context,
    RocalTensor p_input1,
    RocalTensor p_input2,
    bool is_output,
    RocalFloatParam p_stddev,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);

    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    auto stddev = static_cast<FloatParam*>(p_stddev);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<NonLinearBlendNode>({input1, input2}, {output})->init(stddev);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNonLinearBlendFixed(
    RocalContext p_context,
    RocalTensor p_input1,
    RocalTensor p_input2,
    float stddev,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);

    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<NonLinearBlendNode>({input1, input2}, {output})->init(stddev);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalDilate(RocalContext p_context,
            RocalTensor p_input,
            bool is_output,
            unsigned kernel_size,
            RocalTensorLayout output_layout,
            RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<DilateNode>({input}, {output})->init(kernel_size);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalErode(RocalContext p_context,
           RocalTensor p_input,
           bool is_output,
           unsigned kernel_size,
           RocalTensorLayout output_layout,
           RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ErodeNode>({input}, {output})->init(kernel_size);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalMagnitude(RocalContext p_context,
               RocalTensor p_input1,
               RocalTensor p_input2,
               bool is_output,
               RocalTensorLayout output_layout,
               RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);
    auto context = static_cast<Context*>(p_context);
    auto input1  = static_cast<Tensor*>(p_input1);
    auto input2  = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph
            ->add_node<MagnitudeNode>({input1, input2}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPhase(RocalContext p_context,
           RocalTensor p_input1,
           RocalTensor p_input2,
           bool is_output,
           RocalTensorLayout output_layout,
           RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);
    auto context = static_cast<Context*>(p_context);
    auto input1  = static_cast<Tensor*>(p_input1);
    auto input2  = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph
            ->add_node<PhaseNode>({input1, input2}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalThreshold(
    RocalContext p_context,
    RocalTensor p_input,
    std::vector<float> &min,
    std::vector<float> &max,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ThresholdNode>({input}, {output})->init(min, max);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalWarpPerspective(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    unsigned dest_height,
    unsigned dest_width,
    std::vector<float>& perspective,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0) {
            dest_width  = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_dtype);

        // For the warp perspective node, user can create an image with a different width and height
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);

        context->master_graph
            ->add_node<WarpPerspectiveNode>({input}, {output})
            ->init(perspective, static_cast<ResizeInterpolationType>(interpolation_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalErase(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    std::vector<float>& anchor,
    std::vector<float>& shape,
    std::vector<unsigned>& num_boxes,
    std::vector<float>& fill_value,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        if (anchor.size() != shape.size())
            THROW("Anchor and shape dimensions do not match")
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        auto erase_node = context->master_graph->add_node<EraseNode>({input}, {output});
        erase_node->init(anchor, shape, num_boxes, fill_value);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRemap(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    unsigned dest_height,
    unsigned dest_width,
    std::vector<float>& row_remap,
    std::vector<float>& col_remap,
    RocalResizeInterpolationType interpolation_type,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        if (dest_width == 0 || dest_height == 0) {
            dest_width  = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_dtype);
        output_info.modify_dims_width_and_height(op_tensor_layout, dest_width, dest_height);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<RemapNode>({input}, {output})->init(row_remap, col_remap, static_cast<ResizeInterpolationType>(interpolation_type));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropAndPatch(
    RocalContext p_context,
    RocalTensor p_input1,
    RocalTensor p_input2,
    bool is_output,
    std::vector<int>& crop_roi_vec,
    std::vector<int>& patch_roi_vec,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);
    ROCAL_INVALID_INPUT_ERR(p_input2, output);
    auto context = static_cast<Context*>(p_context);
    auto input1  = static_cast<Tensor*>(p_input1);
    auto input2  = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<CropAndPatchNode>({input1, input2}, {output})->init(crop_roi_vec, patch_roi_vec);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRicap(RocalContext p_context,
           RocalTensor p_input,
           bool is_output,
           std::vector<unsigned> &permutation,
           std::vector<int> &crop_rois,
           RocalTensorLayout output_layout,
           RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);

    auto context = static_cast<Context*>(p_context);
    auto input   = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        if (op_tensor_layout != RocalTensorlayout::NONE)
            output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        auto ricap_node = context->master_graph->add_node<RicapNode>({input}, {output});
        ricap_node->init(permutation, crop_rois);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBitwiseOps(RocalContext p_context,
                RocalTensor p_input1,
                RocalTensor p_input2,
                bool is_output,
                RocalBitwiseOp op,
                RocalTensorLayout output_layout,
                RocalTensorOutputType output_datatype) {
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input1, output);

    auto context = static_cast<Context*>(p_context);
    auto input1  = static_cast<Tensor*>(p_input1);
    auto input2  = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensor_layout  = static_cast<RocalTensorlayout>(output_layout);
        RocalTensorDataType op_tensor_dtype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_dtype);
        output = context->master_graph->create_tensor(output_info, is_output);
        auto node = context->master_graph->add_node<BitwiseOpsNode>({input1, input2}, {output});
        node->init(static_cast<BitwiseOp>(op));
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor rocalLog(RocalContext p_context,
                     RocalTensor p_input,
                     bool is_output) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalLog requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        // Preserve FP16 inputs, promote everything else to FP32 to avoid precision loss.
        RocalTensorDataType input_dtype = static_cast<RocalTensorDataType>(input->data_type());
        RocalTensorDataType op_tensor_data_type = (input_dtype == RocalTensorDataType::FP16)
                                                      ? RocalTensorDataType::FP16
                                                      : RocalTensorDataType::FP32;
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_data_type);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<LogNode>({input}, {output});
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalWater(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    RocalFloatParam p_amplitude_x,
    RocalFloatParam p_amplitude_y,
    RocalFloatParam p_frequency_x,
    RocalFloatParam p_frequency_y,
    RocalFloatParam p_phase_x,
    RocalFloatParam p_phase_y,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalWater requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto amplitude_x = static_cast<FloatParam*>(p_amplitude_x);
    auto amplitude_y = static_cast<FloatParam*>(p_amplitude_y);
    auto frequency_x = static_cast<FloatParam*>(p_frequency_x);
    auto frequency_y = static_cast<FloatParam*>(p_frequency_y);
    auto phase_x = static_cast<FloatParam*>(p_phase_x);
    auto phase_y = static_cast<FloatParam*>(p_phase_y);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<WaterNode>({input}, {output})->init(amplitude_x, amplitude_y, frequency_x, frequency_y, phase_x, phase_y);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalWaterFixed(
    RocalContext p_context,
    RocalTensor p_input,
    float amplitude_x,
    float amplitude_y,
    float frequency_x,
    float frequency_y,
    float phase_x,
    float phase_y,
    bool is_output,
    RocalTensorLayout output_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 5)
    THROW("rocalWaterFixed requires vx_rpp version >= 3.1.5");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        if (op_tensor_layout == RocalTensorlayout::NONE)
            op_tensor_layout = input->info().layout();
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<WaterNode>({input}, {output})->init(amplitude_x, amplitude_y, frequency_x, frequency_y, phase_x, phase_y);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorToGreyscale(
    RocalContext p_context,
    RocalTensor p_input,
    bool is_output,
    int subpixel_layout,
    RocalTensorOutputType output_datatype) {
#if !VX_EXT_RPP_CHECK_VERSION(3, 1, 6)
    THROW("rocalColorToGreyscale requires vx_rpp version >= 3.1.6");
    return nullptr;
#endif
    Tensor* output = nullptr;
    ROCAL_INVALID_CONTEXT_ERR(p_context, output);
    ROCAL_INVALID_INPUT_ERR(p_input, output);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_data_type(op_tensor_datatype);
        RocalTensorlayout input_layout = input->info().layout();
        // RPP color_to_greyscale requires output layout to be NCHW with c=1
        if (input_layout == RocalTensorlayout::NHWC) {
            // First convert layout from NHWC to NCHW
            output_info.set_tensor_layout(RocalTensorlayout::NCHW);
            // Now dims are in NCHW format [N, C, H, W], set C=1 for greyscale
            std::vector<size_t> dims = output_info.dims();
            dims[1] = 1;
            output_info.set_dims(dims);
        } else if (input_layout == RocalTensorlayout::NCHW) {
            std::vector<size_t> dims = output_info.dims();
            dims[1] = 1;
            output_info.set_dims(dims);
        } else {
            THROW("ColorToGreyscale only supports NHWC or NCHW layouts")
        }
        output_info.set_color_format(RocalColorFormat::U8);
        output = context->master_graph->create_tensor(output_info, is_output);
        ColorToGreyscaleNode::SubpixelLayout layout = static_cast<ColorToGreyscaleNode::SubpixelLayout>(subpixel_layout);
        context->master_graph->add_node<ColorToGreyscaleNode>({input}, {output})->init(layout);
    } catch (const std::exception& e) {
        ROCAL_PRINT_EXCEPTION(context, e);
    }
    return output;
}
