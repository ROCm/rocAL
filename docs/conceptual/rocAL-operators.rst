.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation, operators, augmentations, readers, decoders

.. _overview:

********************************************************************
rocAL Operators
********************************************************************

rocAL operators offer the flexibility to run on CPU or GPU for building hybrid pipelines. They also support classification and object detection on the workload. Some of the useful operators supported by rocAL are listed below:

* **Augmentations:** These are used to enhance the data set by adding effects to the original images. 
  To use the augmentations, import the instance of ``amd.rocal.fn`` into the Python script. These augmentation 
  APIs further call the RPP kernels underneath (HIP/HOST) depending on the backend used to build RPP and rocAL.
  For the complete list of augmentations, see the :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>`. 
  For the parameters of each augmentation, see the :doc:`augmentations Python API reference <../doxygen/html/fn_8py>`.

* **Readers:** These are used to read and understand the different types of datasets and their metadata. Some 
  examples of readers are list of files with folders, LMDB, TFRecord, and JSON file for metadata. To use the 
  readers, import the instance of ``amd.rocal.readers`` into the Python script. 
  For the parameters of each reader, see the :doc:`readers Python API reference <../doxygen/html/readers_8py>`.

* **Decoders:** These are used to support different input formats of images and videos. Decoders extract 
  data from the datasets that are in compressed formats such as JPEG, MP4, etc. To use the decoders, 
  import the instance of ``amd.rocal.decoders`` into the Python script. 
  For the parameters of each decoder, see the :doc:`decoders Python API reference <../doxygen/html/decoders_8py>`.

Augmentations
=============

Augmentations are grouped by the kind of change they make to the data. Table 1 summarizes each group and lists some of its augmentations. 
The :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>` describes every augmentation in each group. 
To learn how to add augmentations to a pipeline, see :doc:`Using augmentations in a rocAL pipeline <../how-to/rocAL-use-augmentations>`.

.. list-table:: Table 1. Augmentation groups available through rocAL
  :header-rows: 1
  :widths: 20 45 35

  * - Group
    - Description
    - Examples
  * - Color
    - Change the color, brightness, contrast, or tone of an image without changing its shape or size.
    - Blend, Brightness, Color Temperature, Color Twist, Contrast, Exposure, Gamma, Hue, Saturation, Vignette
  * - Effects
    - Add visual effects, noise, or degradation to an image. Use them to make a model robust to imperfect input.
    - Fog, Jitter, Pixelization, Rain, Snow, Salt and Pepper Noise, Gaussian Noise, Shot Noise, Water
  * - Geometry
    - Change the size, position, orientation, or shape of an image, or select a region of it.
    - Crop, Crop Mirror Normalization, Crop Resize, Fisheye Lens, Flip, Lens Correction, Random Crop, Resize, Resize Crop Mirror, Rotation, Warp Affine, Warp Perspective
  * - Volumetric and tensor
    - Operate on multidimensional tensors, such as flipping along the depth dimension or slicing and cropping a region of a 3D volume. 
    - Flip (depth), Slice, ROI Random Crop, Random Object Bounding Box
  * - Audio
    - Prepare audio signals for training and inference.
    - Preemphasis Filter, Spectrogram, Mel Filter Bank, To Decibels, Resample, Nonsilent Region
  * - Math and utility
    - Perform calculations on tensors, prepare labels, copy data, or add your own code to a pipeline.
    - Normalize, Transpose, Tensor Sum, One Hot, Copy, Python Function

Readers
=======

Readers load data and its metadata from a dataset and pass it to a decoder. Table 2 lists the readers in ``amd.rocal.readers``. 
For the complete parameter lists, see the :doc:`readers Python API reference <../doxygen/html/readers_8py>`.

.. list-table:: Table 2. Readers available through rocAL
  :header-rows: 1
  :widths: 25 25 50

  * - Reader
    - Python function
    - Description
  * - File Reader
    - ``readers.file``
    - Reads files, such as images or audio, from a list of files in a folder or folders, or from a file list. Returns each file with its label.
  * - Video Reader
    - ``readers.video``
    - Reads and decodes video from a list of files in a folder or folders, and returns the video as sequences of frames. Optionally returns the frame numbers and timestamps.
  * - Video Reader with resize
    - ``readers.video_resize``
    - Reads and decodes video in the same way as the video reader, and resizes the frames.
  * - Sequence Reader
    - ``readers.sequence_reader``
    - Reads sequences of image files from a folder.
  * - Caffe LMDB Reader
    - ``readers.caffe``
    - Reads (key, value) pairs from Caffe LMDB.
  * - Caffe2 LMDB Reader
    - ``readers.caffe2``
    - Reads (key, value) pairs from Caffe2 LMDB.
  * - COCO Reader - file source and keypoints
    - ``readers.coco``
    - Reads images and JSON annotations from COCO dataset.
  * - TFRecord Reader
    - ``readers.tfrecord``
    - Reads from a TFRecord dataset.
  * - MXNet Reader
    - ``readers.mxnet``
    - Reads from a RecordIO dataset.
  * - Web Dataset Reader
    - ``readers.webdataset``
    - Reads from a web dataset. The data is stored in ``.tar`` files.
  * - NumPy Reader
    - ``readers.numpy``
    - Reads data from NumPy (``.npy``) files.
  * - CIFAR-10 Dataset Reader
    - ``readers.cifar10``
    - Reads from a binary CIFAR-10 dataset.

To read data from a source that isn't in a supported dataset format, use the external source reader, ``fn.external_source``. 
It reads data from an iterator or callable that you provide. 

Decoders
========

Decoders convert compressed data from the readers into data that can be processed by a pipeline. Table 3 lists the decoders in ``amd.rocal.decoders``. 
For the complete parameter lists, see the :doc:`decoders Python API reference <../doxygen/html/decoders_8py>`.

.. list-table:: Table 3. Decoders available through rocAL
  :header-rows: 1
  :widths: 25 25 50

  * - Decoder
    - Python function
    - Description
  * - Image
    - ``decoders.image``
    - Decodes JPEG images. The decoder uses TurboJPEG by default. When ``device="gpu"``, the decoder uses rocJPEG to run the decoding on the GPU. 
  * - Image_raw
    - ``decoders.image_raw``
    - Decodes images in raw format. Use this decoder with the TFRecord reader.
  * - Image_random_crop
    - ``decoders.image_random_crop``
    - Decodes and randomly crops JPEG images. The cropping window is chosen based on a range of aspect ratios and areas.
  * - Image_slice
    - ``decoders.image_slice``
    - Decodes and slices JPEG images. 
  * - Audio
    - ``decoders.audio``
    - Decodes WAV audio files.

Video files are decoded by the video readers, ``readers.video`` and ``readers.video_resize``, so there's no separate video decoder.

To see examples demonstrating the usage of decoders and readers, see 
`rocAL Python Examples <https://github.com/ROCm/rocAL/tree/develop/docs/examples>`_.
