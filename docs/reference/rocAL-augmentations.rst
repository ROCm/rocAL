.. meta::
  :description: rocAL augmentations reference
  :keywords: rocAL, ROCm, API, augmentations, operators, reference

.. _augmentations-reference:

********************************************************************
rocAL augmentations reference
********************************************************************

This topic lists the augmentations and related operators that rocAL provides in the ``amd.rocal.fn`` module, including the ``fn.random`` random number generators, grouped by the kind of change they make to the data. 
Use it to find the augmentation you need, then see the :doc:`augmentations Python API reference <../doxygen/html/fn_8py>` for the complete list of parameters, their default values, and their types.

To learn how to add augmentations to a pipeline, see :doc:`Using augmentations in a rocAL pipeline <../how-to/rocAL-use-augmentations>`. 
For an overview of the readers and decoders that supply data to the augmentations, see :doc:`rocAL operators <../conceptual/rocAL-operators>`.

Augmentations are called through ``amd.rocal.fn``:

.. code:: python

  import amd.rocal.fn as fn

Unless noted otherwise, the augmentations take the output of a decoder or of another augmentation as their first argument and return the augmented data. 
Most image augmentations also accept ``output_layout`` and ``output_dtype`` arguments that set the layout and data type of the output tensor.

.. note::

  The ``device`` argument that most image augmentations accept is unused. The augmentation runs on the backend that was selected when the pipeline was created.

Color augmentations
===================

Color augmentations change the color, brightness, contrast, or tone of an image without changing its size.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Augmentation
    - Description
  * - ``fn.blend``
    - Blends two images using a ratio: ``output = input1 * ratio + input2 * (1 - ratio)``.
  * - ``fn.brightness``
    - Adjusts the brightness of an image using a brightness multiplier and a brightness shift. A multiplier of 0 gives a black image, 1 leaves the image unchanged, and 2 doubles the brightness. The ``conditional_execution`` argument controls whether the augmentation runs.
  * - ``fn.brightness_fixed``
    - Adjusts the brightness of an image using fixed values for the multiplier and the shift. The ``conditional_execution`` argument controls whether the augmentation runs.
  * - ``fn.color_cast``
    - Applies a color cast by blending a target RGB color with the input using an alpha value.
  * - ``fn.color_jitter``
    - Applies color jitter to an image. Use the ``brightness``, ``contrast``, ``hue``, and ``saturation`` arguments to set the adjustments. This augmentation is supported on the CPU backend only. On a GPU pipeline, the call raises a ``RuntimeError``.
  * - ``fn.color_temp``
    - Adjusts the color temperature of an image.
  * - ``fn.color_twist``
    - Adjusts the brightness, contrast, hue, and saturation of an image in one operation.
  * - ``fn.color_to_greyscale``
    - Converts a color image to greyscale.
  * - ``fn.channel_permute``
    - Permutes the channels of an image.
  * - ``fn.contrast``
    - Adjusts the contrast of an image.
  * - ``fn.exposure``
    - Adjusts the exposure of an image.
  * - ``fn.gamma_correction``
    - Applies gamma correction to an image.
  * - ``fn.hue``
    - Adjusts the hue of an image.
  * - ``fn.saturation``
    - Adjusts the saturation of an image.
  * - ``fn.vignette``
    - Applies a vignette effect to an image.
  * - ``fn.lut``
    - Applies a look-up table (LUT) transformation to an image. The LUT is created internally and inverts the image: ``output[i] = 255 - input[i]`` for 8-bit data.
  * - ``fn.posterize``
    - Reduces the color channels of an image to ``num_bits`` bits.
  * - ``fn.solarize``
    - Applies a solarize effect to an image. The ``threshold`` value is in the range 0.0 to 1.0.
  * - ``fn.threshold``
    - Applies thresholding to an image, using ``min`` and ``max`` threshold values for each channel or for each sample.
  * - ``fn.bitwise_ops``
    - Applies a bitwise operation to images. The ``op`` argument selects the operation: ``BITWISE_AND``, ``BITWISE_OR``, ``BITWISE_XOR``, or ``BITWISE_NOT``. ``BITWISE_NOT`` uses only the first input.
  * - ``fn.non_linear_blend``
    - Blends two images non-linearly, using a standard deviation value for each sample.

Effects augmentations
=====================

Effects augmentations add visual effects, noise, or degradation to an image. Use them to make a model more robust to imperfect input.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Augmentation
    - Description
  * - ``fn.blur``
    - Applies a blur effect to an image. Use ``window_size`` and ``sigma`` to control the blur.
  * - ``fn.fog``
    - Applies a fog effect to an image.
  * - ``fn.jitter``
    - Applies a jitter effect to an image.
  * - ``fn.pixelate``
    - Applies a pixelate effect to an image.
  * - ``fn.rain``
    - Applies a rain effect to an image.
  * - ``fn.snow``
    - Applies a snow effect to an image.
  * - ``fn.spatter``
    - Applies a spatter effect with a specified color to an image.
  * - ``fn.water``
    - Applies a water effect to an image.
  * - ``fn.gaussian_noise``
    - Adds Gaussian noise to an image. The ``conditional_execution`` argument controls whether the augmentation runs.
  * - ``fn.shot_noise``
    - Adds shot (Poisson) noise to an image.
  * - ``fn.snp_noise``
    - Adds salt-and-pepper noise to an image.
  * - ``fn.gaussian_filter``
    - Applies a Gaussian filter to an image, using a standard deviation value for each sample.
  * - ``fn.median_filter``
    - Applies a median filter to an image.
  * - ``fn.dilate``
    - Applies morphological dilation to an image.
  * - ``fn.erode``
    - Applies morphological erosion to an image.
  * - ``fn.jpeg_compression_distortion``
    - Applies JPEG compression distortion to an image. The ``quality`` argument sets the compression quality.
  * - ``fn.grid_mask``
    - Applies a GridMask effect by overlaying a grid of tiles with a masked ratio.
  * - ``fn.erase``
    - Erases regions of an image. Specify the boxes as vectors: ``anchor`` is the top-left corner (x1, y1) of each box, and ``shape`` is the bottom-right corner (x2, y2) of each box, inclusive. Despite its name, ``shape`` isn't a width and height. Getting the boxes from tensors using the ``anchor_box_info`` and ``colors`` arguments isn't implemented.
  * - ``fn.crop_and_patch``
    - Crops a region from the second input and patches it into a region of the first input. The regions are specified in XYWH format: ``[x, y, w, h]``.
  * - ``fn.ricap``
    - Applies RICAP (Random Image Cropping And Patching). RICAP creates a new training image by combining four cropped regions from the input batch.

Geometry augmentations
======================

Geometry augmentations change the size, position, orientation, or shape of an image, or select a region of it.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Augmentation
    - Description
  * - ``fn.crop``
    - Crops an image. Specify the position of the crop window and its size (``crop_w`` and ``crop_h``, and ``crop_d`` for a 3D input).
  * - ``fn.center_crop``
    - Crops an image at its center.
  * - ``fn.random_crop``
    - Crops an image randomly. The crop window is generated from the ``crop_area_factor`` and ``crop_aspect_ratio`` ranges. The ``num_attempts`` argument sets how many times rocAL tries to find a window that meets those conditions.
  * - ``fn.ssd_random_crop``
    - Crops an image randomly, for SSD training. The ``p_threshold`` argument sets the threshold used to select bounding boxes during crop generation.
  * - ``fn.random_bbox_crop``
    - Crops an image randomly, based on the bounding boxes of the objects in the image. Use the ``aspect_ratio``, ``scaling``, and ``crop_shape`` arguments to control the crop windows, and ``all_boxes_above_threshold`` to require that all bounding boxes in a sample overlap with the crop window.
  * - ``fn.crop_mirror_normalize``
    - Crops, normalizes, and flips an image in one operation.
  * - ``fn.resize``
    - Resizes an image. Specify the output size using ``resize_width`` and ``resize_height``, or use ``resize_shorter`` or ``resize_longer`` to set the length of one dimension and keep the aspect ratio. Use ``interpolation_type`` to select the interpolation method.
  * - ``fn.resize_crop``
    - Resizes and crops an image in one operation.
  * - ``fn.resize_crop_mirror``
    - Resizes, crops, and flips an image in one operation.
  * - ``fn.resize_mirror_normalize``
    - Resizes, normalizes, and flips an image in one operation.
  * - ``fn.random_resized_crop``
    - Crops a random portion of an image and resizes it to the given size in one operation.
  * - ``fn.roi_resize``
    - Resizes an image based on a region of interest (ROI).
  * - ``fn.flip``
    - Flips an image horizontally, vertically, along the depth dimension, or along a combination of the three. The ``depth`` argument applies to 3D inputs.
  * - ``fn.rotate``
    - Rotates an image.
  * - ``fn.fish_eye``
    - Applies a fish-eye lens effect to an image.
  * - ``fn.lens_correction``
    - Applies lens correction to an image, using a camera matrix and distortion coefficients.
  * - ``fn.warp_affine``
    - Applies an affine transformation to an image.
  * - ``fn.warp_perspective``
    - Applies a perspective transformation to an image.
  * - ``fn.remap``
    - Remaps the pixels of an image, using a row map and a column map for each sample.

Volumetric and tensor augmentations
===================================

These augmentations operate on multidimensional tensors, such as 3D volumes. 

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Augmentation
    - Description
  * - ``fn.slice``
    - Extracts a slice of the input tensor. Specify the slice using its starting coordinates (``anchor``) and its ``shape``. The ``out_of_bounds_policy`` argument determines what happens when the slice extends outside the input: ``"error"``, ``"pad"``, or ``"trim_to_shape"``. The ``fill_values`` argument sets the padding values when the policy is ``"pad"``.
  * - ``fn.roi_random_crop``
    - Generates a random crop anchor within a region of interest (ROI). The ROI is defined by ``roi_start`` and ``roi_end``. The anchor is chosen so that a crop of the given ``crop_shape`` fits within the ROI and within the bounds of the input tensor. Returns a tensor that contains the anchor coordinates for each sample in the batch.
  * - ``fn.random_object_bbox``
    - Finds the bounding boxes of the connected components (objects) in a segmentation mask and returns the bounding box of one randomly selected object. The ``format`` argument sets how the box is returned: ``"anchor_shape"`` (default) returns anchor and shape tensors, ``"start_end"`` returns start and end tensors, and ``"box"`` returns a single tensor with the concatenated start and end coordinates. The input tensor must have at least four dimensions, for example NCDHW layout. Otherwise, the operator throws an exception.

Use ``fn.roi_random_crop`` or ``fn.random_object_bbox`` to generate the ``anchor`` for ``fn.slice``. 

Audio augmentations
===================

Audio augmentations prepare audio signals for training and inference. Decode the audio with ``fn.decoders.audio`` first. See :doc:`rocAL operators <../conceptual/rocAL-operators>` for the decoders.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Augmentation
    - Description
  * - ``fn.preemphasis_filter``
    - Applies a preemphasis filter to the audio signal: ``Y[t] = X[t] - coeff * X[t-1]``. The value of ``X[t-1]`` at ``t = 0`` is set by the ``border`` argument: ``"zero"``, ``"clamp"``, or ``"reflect"``.
  * - ``fn.spectrogram``
    - Produces a spectrogram from a 1D signal. The input must be a single channel, with a shape of ``(srcLength, 1)`` and the ``float32`` data type.
  * - ``fn.mel_filter_bank``
    - Converts a spectrogram to a mel spectrogram by applying a bank of triangular filters.
  * - ``fn.to_decibels``
    - Converts a magnitude (real, positive) to the decibel scale.
  * - ``fn.resample``
    - Resamples an audio signal by applying a sinc filter with a Hann window. The ``quality`` argument controls the extent of the window.
  * - ``fn.nonsilent_region``
    - Detects the leading and trailing silence in an audio buffer. Returns the anchor and shape of the non-silent region.

Math and tensor operators
=========================

These operators perform calculations on tensors and are typically used between other operators in a pipeline.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Operator
    - Description
  * - ``fn.normalize``
    - Normalizes the input by removing the mean and dividing by the standard deviation: ``out = scale * (in - mean) / stddev + shift``. The mean and standard deviation can be calculated for the axes you specify, or you can provide them using the ``mean`` and ``stddev`` arguments.
  * - ``fn.transpose``
    - Transposes the input according to the permutation you pass in ``perm``.
  * - ``fn.log``
    - Computes the natural logarithm of each input element.
  * - ``fn.log1p``
    - Computes the natural logarithm of 1 + each input element.
  * - ``fn.magnitude``
    - Computes the magnitude of complex input images.
  * - ``fn.phase``
    - Computes the phase of complex input images.
  * - ``fn.tensor_add_tensor_float``
    - Adds one tensor to another.
  * - ``fn.tensor_mul_scalar_float``
    - Multiplies a tensor by a scalar float value.
  * - ``fn.tensor_sum``
    - Computes the sum of the tensor elements for each image.
  * - ``fn.tensor_min``
    - Computes the minimum of the tensor elements for each image.
  * - ``fn.tensor_max``
    - Computes the maximum of the tensor elements for each image.
  * - ``fn.tensor_mean``
    - Computes the mean of the tensor elements for each image.
  * - ``fn.tensor_stddev``
    - Computes the standard deviation of the tensor elements for each image, using precomputed means that you pass in ``mean_tensor``.

Utility operators
=================

Utility operators help to prepare labels, copy data, extend pipelines with your own code, and support object detection.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Operator
    - Description
  * - ``fn.one_hot``
    - Applies one-hot encoding to labels. Labels are zero-based: a sample labeled ``k`` is encoded with a 1 in column ``k`` and a 0 elsewhere. A label outside the range ``[0, num_classes)`` can't be encoded and raises an error when the batch is read.
  * - ``fn.box_encoder``
    - Encodes bounding boxes against a set of anchors, which are given in ltrb format. The ``criteria`` argument sets the threshold IoU (intersection over union) for matching bounding boxes with anchors.
  * - ``fn.box_iou_matcher``
    - Matches boxes with anchors using IoU, with a ``high_threshold`` and a ``low_threshold``. Returns the matched boxes and the list of matched indices.
  * - ``fn.copy``
    - Copies the input tensor to the output tensor.
  * - ``fn.nop``
    - Performs no operation. The returned tensor is allocated but never written, so its contents are undefined and can differ between runs and between backends. The output isn't a copy of the input. Use ``fn.nop`` only to measure pipeline overhead. Use ``fn.copy`` when the output must carry the input data.
  * - ``fn.python_function``
    - Calls a Python callable that you supply on the whole batch. This operator runs on the host (CPU) only. The input is exposed as a NumPy view of the batch, and the callable must return a NumPy array with a matching batch size. This operator is disabled by default. It raises a ``RuntimeError`` unless rocAL was built with ``-DROCAL_PYTHON_FUNCTION=ON``.
  * - ``fn.external_source``
    - Reads data from an iterator or a callable that you supply. The ``mode`` argument sets the type of data the source provides: file names (the default), raw compressed data, or raw uncompressed data. 

Random number generation
========================

Random number generators produce values that you can pass as arguments to augmentations, so that each sample in a batch gets a different value.

.. list-table:: 
  :header-rows: 1
  :widths: 28 72

  * - Function
    - Description
  * - ``fn.uniform``
    - Generates random float values from a uniform distribution within a ``range``, for example ``fn.uniform(range=[0.875, 1.125])``. Pass the result to an augmentation argument such as ``brightness``.
  * - ``fn.random.coin_flip``
    - Generates 0 or 1 values using a given ``probability`` of getting a 1. Use it, for example, as the ``mirror`` argument of ``fn.crop_mirror_normalize``.
  * - ``fn.random.uniform``
    - Creates a tensor from a uniform distribution with a given ``range``. 
  * - ``fn.random.normal``
    - Creates a tensor from a normal distribution with a given ``mean`` and ``stddev``.
