.. meta::
  :description: rocAL Jupyter notebooks
  :keywords: rocAL, ROCm, API, examples, Jupyter, notebooks

.. _notebooks:

********************************************************************
rocAL Jupyter notebooks
********************************************************************

The rocAL repository includes Jupyter notebooks that walk through how to build rocAL pipelines for different kinds of data and frameworks. 
Each notebook is a self-contained example that you can run, read, and adapt. 
The notebooks are in the `docs/examples/notebooks <https://github.com/ROCm/rocAL/tree/develop/docs/examples/notebooks>`_ folder of the rocAL repository.

This topic describes what each notebook demonstrates, which rocAL operators it uses, and what you need to run it.

Before you run a notebook
=========================

To run the notebooks, you need:

* rocAL installed with its Python module. See :doc:`rocAL prerequisites <../install/rocAL-prerequisites>` and the rocAL installation topics.
* A Jupyter environment, such as JupyterLab or Jupyter Notebook.
* The Python packages that the notebook imports, such as ``matplotlib``. Some notebooks need a framework, such as PyTorch, TensorFlow, or JAX, or another package, such as ``librosa``. Table 1 lists these requirements.
* The data that the notebook reads. Table 1 lists the data that each notebook needs.

Many notebooks read sample data from a folder structure named ``rocal_data``. For these notebooks, set the ``ROCAL_DATA_PATH`` environment variable to the directory that contains the ``rocal_data`` folder before you start Jupyter. 
Some notebooks stop with an error if the variable isn't set. 

Other notebooks set the path to the data in a variable in the notebook, such as ``data_path`` or ``image_dir``. 
Update the variable if your data is in a different location. 
The image notebooks ``augmentation_examples``, ``decoder_examples``, ``python_function_example``, and ``resize_implementation`` default to the sample images in the ``share/rocal/test/data/images/AMD-tinyDataSet/`` folder of the ROCm installation, which is ``/opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet/`` in a default installation.

.. list-table:: Table 1. rocAL Jupyter notebooks
  :header-rows: 1
  :widths: 24 36 40

  * - Notebook
    - What it demonstrates
    - Requirements
  * - :ref:`audio_decoder <notebook-audio-decoder>`
    - Loading and decoding WAV audio files.
    - PyTorch, ``matplotlib``, ``ROCAL_DATA_PATH`` with audio data
  * - :ref:`augmentation_examples <notebook-augmentation-examples>`
    - Applying several augmentations to the same images and comparing the results.
    - ``matplotlib``, a folder of images
  * - :ref:`classification_training_flowerdataset <notebook-classification-training>`
    - Training an image classification model with PyTorch and rocAL.
    - PyTorch, torchvision, the flower dataset created by ``create_classification_flower_dataset.py``
  * - :ref:`decoder_examples <notebook-decoder-examples>`
    - Decoding images on the CPU and on the GPU.
    - ``matplotlib``, a GPU that rocJPEG supports for the GPU examples, a folder of images
  * - :ref:`jax_training_example <notebook-jax-training>`
    - Training a model on multiple GPUs with JAX and rocAL.
    - JAX, more than one GPU for multi-GPU training, CIFAR-10 binary data
  * - :ref:`python_function_example <notebook-python-function>`
    - Adding custom Python code to a pipeline.
    - rocAL built with ``-DROCAL_PYTHON_FUNCTION=ON``, ``matplotlib``, a folder of images
  * - :ref:`resize_implementation <notebook-resize-implementation>`
    - Comparing the interpolation types of the resize augmentation.
    - ``matplotlib``, a folder of images
  * - :ref:`spectrogram <notebook-spectrogram>`
    - Calculating an audio spectrogram and comparing it to a reference implementation.
    - PyTorch, ``librosa``, ``matplotlib``, ``ROCAL_DATA_PATH`` with audio data
  * - :ref:`tf_dataloader <notebook-tf-dataloader>`
    - Reading data stored in the TFRecord format.
    - TensorFlow, ``matplotlib``, ``ROCAL_DATA_PATH`` with TFRecord data
  * - :ref:`video_decoder <notebook-video-decoder>`
    - Reading and decoding video from multiple files.
    - PyTorch, ``matplotlib``, ``ROCAL_DATA_PATH`` with video data
  * - :ref:`video_reader_with_label <notebook-video-reader-with-label>`
    - Reading and decoding video along with its labels.
    - PyTorch, ``matplotlib``, ``ROCAL_DATA_PATH`` with video data
  * - :ref:`webdataset_reader_example <notebook-webdataset-reader>`
    - Reading data stored in the WebDataset format.
    - ``matplotlib``, ``ROCAL_DATA_PATH`` with WebDataset data

For information about the operators that the notebooks use, see :doc:`rocAL operators <../conceptual/rocAL-operators>` and the :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>`. 
For information about creating and running pipelines, see :doc:`Create and run the pipeline <../how-to/rocAL-use-pipeline>`.

.. _notebook-audio-decoder:

audio_decoder
=============

This notebook shows how to set up a simple audio decoder pipeline. It loads WAV audio files and their labels, decodes the audio, and plots the first decoded audio sample.

* **Operators:** ``fn.readers.file`` reads the audio files and their labels from a file list. ``fn.decoders.audio`` decodes the audio, using the ``file_root`` and ``file_list_path`` arguments.
* **Iterator:** ``ROCALAudioIterator``, from ``amd.rocal.plugin.pytorch``. The iterator returns the audio data as PyTorch tensors, the labels, and the region of interest (ROI).
* **Pipeline configuration:** The notebook uses a batch size of 1 and the CPU backend (``rocal_cpu=True``) to keep the example simple.
* **Data:** Set ``ROCAL_DATA_PATH``. The notebook reads the audio from ``${ROCAL_DATA_PATH}/rocal_data/audio/``. This folder must contain the ``.wav`` files and a ``wav_file_list.txt`` file that lists them.

.. _notebook-augmentation-examples:

augmentation_examples
=====================

This notebook shows a simple pipeline that applies several augmentations to the same images. It displays the original resized image and the result of each augmentation side by side.

* **Operators:** ``fn.readers.file`` and ``fn.decoders.image`` read and decode the images. ``fn.resize`` resizes them. The notebook then applies ``fn.brightness``, ``fn.contrast``, ``fn.fish_eye``, ``fn.fog``, ``fn.exposure``, ``fn.color_twist``, and ``fn.saturation`` to the resized images.
* **Outputs:** The pipeline is created with the ``Pipeline()`` constructor, and ``pipe.set_outputs()`` returns the output of every augmentation.
* **Iterator:** ``ROCALClassificationIterator``, from ``amd.rocal.plugin.generic``. The notebook displays the outputs with ``imshow()``.
* **Data:** Set ``data_path`` to a folder of images. 

To learn more about using augmentations, see :doc:`Using augmentations in a rocAL pipeline <../how-to/rocAL-use-augmentations>`.

.. _notebook-classification-training:

classification_training_flowerdataset
=====================================

This notebook shows an image classification training workflow with PyTorch, using a small flower dataset. 
It defines rocAL pipelines for training and validation, trains a model, saves checkpoints, and tests the model with a batch of images from the test set.

* **Training pipeline:** ``fn.readers.file`` reads the images. ``fn.decoders.image_random_crop`` decodes and randomly crops them. ``fn.resize`` resizes them to 224 x 224 using triangular interpolation, and ``fn.crop_mirror_normalize`` normalizes them and randomly flips them using ``fn.random.coin_flip``. The pipeline returns (image, label) pairs.
* **Validation pipeline:** The validation pipeline is the same as the training pipeline, but without the random shuffle and the random flip.
* **Training:** The notebook trains a pretrained ResNet-18 model from ``torchvision``. It uses the cross-entropy loss function and the SGD optimizer with momentum, which is wrapped by an implementation of the Lookahead optimizer. It saves a checkpoint each time the validation accuracy improves. 
* **Iterator:** ``ROCALClassificationIterator``, from ``amd.rocal.plugin.pytorch``.
* **Data:** Create the dataset before you run the notebook. Run the ``create_classification_flower_dataset.py`` script that's in the same folder. The script downloads the Oxford 102 Flowers dataset and its labels, keeps the images of the first 50 classes, and splits them into training, validation, and test sets (70%, 20%, and 10%) in ``./Flower102/split_data``. The script installs the ``split_folders`` package, uses ``wget`` to download the dataset, and needs the ``scipy`` and ``pandas`` packages.

To learn more about training with PyTorch, see :doc:`Run PyTorch training with rocAL <../how-to/rocAL-pytorch-framework>`.

.. _notebook-decoder-examples:

decoder_examples
================

This notebook shows the image decoders running on the CPU backend and on the GPU backend. It shows that the decoders offer the same functionality on both backends. 
Each example takes a ``device`` parameter, so you can change the backend by changing one parameter.

* **Operators:** ``fn.decoders.image`` decodes JPEG images. ``fn.decoders.image_slice`` decodes JPEG images and slices them. The notebook resizes the output with ``fn.resize``.
* **GPU decoding:** On the GPU backend, ``fn.decoders.image`` uses rocJPEG to run the JPEG decoding on the built-in hardware decoder of the GPU, if the GPU supports it.
* **Iterator:** ``ROCALClassificationIterator``, from ``amd.rocal.plugin.generic``. 
* **Data:** The notebook reads the sample images in the ``share/rocal/test/data/images/AMD-tinyDataSet/`` folder of the ROCm installation. Set ``image_dir`` to use different images.

.. _notebook-jax-training:

jax_training_example
====================

This notebook shows how to build a GPU-accelerated data loading pipeline for the CIFAR-10 dataset with rocAL and how to use it with JAX for multi-GPU training. 
It demonstrates two JAX parallelization strategies: automatic parallelization using sharding, and explicit data parallelism using ``pmap``.

* **Pipelines:** The notebook creates one training pipeline for each GPU. Each pipeline reads a unique shard of the dataset using the ``shard_id`` and ``num_shards`` arguments of ``fn.readers.cifar10``, so each GPU sees different data. It then applies ``fn.crop_mirror_normalize``. A single validation pipeline reads the whole test dataset.
* **Iterator:** ``ROCALJaxIterator``, from ``amd.rocal.plugin.jax``. For automatic parallelization, the iterator is initialized with a ``sharding`` configuration, so that the data is already split and placed on the correct GPUs.
* **Model:** A simple multi-layer perceptron (MLP) that classifies the CIFAR-10 images.
* **Sharding and ``pmap``:** With sharding, the training loop is almost the same as a single-GPU loop. With ``pmap``, you replicate the model on each device and reshape each global batch so that its first dimension is the number of devices.
* **Data:** The notebook reads the CIFAR-10 binary data from ``<data_path>/cifar-10-batches-bin``. Update ``data_path`` to the location of your data.

To learn more about training with JAX, see :doc:`Run JAX training with rocAL <../how-to/rocAL-jax-framework>`.

.. _notebook-python-function:

python_function_example
=======================

This notebook shows how to use ``fn.python_function`` to add custom Python processing to a rocAL pipeline. 
It builds a simple image pipeline and applies Python-based brightness, crop, and horizontal flip steps. It then shows how to pass a callable class to normalize the batch.

* **Operators:** ``fn.readers.file`` and ``fn.decoders.image`` read and decode the images. The notebook then chains ``fn.python_function`` calls to apply a random brightness, crop the images to 224 x 224, and optionally flip them horizontally.
* **Output:** The notebook displays the flipped output (``uint8``) and the normalized output (``float``) for the same sample.
* **Iterator:** ``ROCALClassificationIterator``, from ``amd.rocal.plugin.generic``.

.. note::

  ``fn.python_function`` is disabled by default. It raises a ``RuntimeError`` unless rocAL was built with ``-DROCAL_PYTHON_FUNCTION=ON``. 
  Building rocAL with this option embeds the CPython C-API in ``librocal.so``. The Python module works, but C and C++ applications fail to link or load ``librocal.so``. 
  For details, see `issue #522 <https://github.com/ROCm/rocAL/issues/522>`_.

.. _notebook-resize-implementation:

resize_implementation
=====================

This notebook shows a simple pipeline that applies the resize augmentation with different interpolation types and displays the results.

* **Operators:** ``fn.readers.file`` and ``fn.decoders.image`` read and decode the images. ``fn.resize`` resizes them using each of the following interpolation types: ``types.NEAREST_NEIGHBOR_INTERPOLATION``, ``types.LINEAR_INTERPOLATION``, ``types.CUBIC_INTERPOLATION``, ``types.GAUSSIAN_INTERPOLATION``, ``types.LANCZOS_INTERPOLATION``, and ``types.TRIANGULAR_INTERPOLATION``.
* **Iterator:** ``ROCALClassificationIterator``, from ``amd.rocal.plugin.generic``. 
* **Data:** Set ``data_path`` to a folder of images. 

.. _notebook-spectrogram:

spectrogram
===========

This notebook shows a pipeline that loads and decodes audio data and calculates its spectrogram. It shows how to create a pipeline, set its outputs, build and run it, and enumerate the results. 
To verify the rocAL implementation, the notebook calculates the same spectrogram with ``librosa``, plots both results, and checks that the numerical difference between them is insignificant.

* **Operators:** ``fn.readers.file`` reads the audio files, ``fn.decoders.audio`` decodes them, and ``fn.spectrogram`` calculates the spectrogram.
* **Iterator:** ``ROCALAudioIterator``, from ``amd.rocal.plugin.pytorch``.
* **Data:** Set ``ROCAL_DATA_PATH``. The notebook reads the audio from ``${ROCAL_DATA_PATH}/rocal_data/audio/``. This folder must contain the ``.wav`` files and a ``wav_file_list.txt`` file that lists them. The ``librosa`` reference calculation uses the sample file ``wav/19-198-0000.wav`` in the same folder. 

To learn about the audio augmentations that rocAL provides, see the :doc:`rocAL augmentations reference <../reference/rocAL-augmentations>`.

.. _notebook-tf-dataloader:

tf_dataloader
=============

This notebook shows how to use rocAL to read data that is saved in the TensorFlow TFRecord format. It resizes the decoded images and displays them.

* **Operators:** ``fn.readers.tfrecord`` reads the TFRecord dataset. ``fn.decoders.image`` decodes the images, ``fn.resize`` resizes them, and ``fn.one_hot`` encodes the labels. 
* **Iterator:** ``ROCALIterator``, from ``amd.rocal.plugin.tf``.
* **Data:** Set ``ROCAL_DATA_PATH``. The notebook reads the TFRecord data from ``${ROCAL_DATA_PATH}/rocal_data/tf/classification``.

To learn more about training with TensorFlow, see :doc:`Run TensorFlow training with rocAL <../how-to/rocAL-tensorflow-framework>`.

.. _notebook-video-decoder:

video_decoder
=============

This notebook shows a simple video pipeline that reads and decodes video data from multiple files, and plots the decoded frame sequences.

* **Operators:** ``fn.readers.video`` reads and decodes the video as sequences of frames. The notebook sets the sequence length to 3. ``fn.crop_mirror_normalize`` processes the sequences.
* **Iterator:** The notebook defines a custom ``ROCALVideoIterator`` class to iterate through the video pipeline outputs. The notebook plots the sequences using ``matplotlib``.
* **Data:** Set ``ROCAL_DATA_PATH``. The notebook reads the videos from ``${ROCAL_DATA_PATH}/rocal_data/video_and_sequence_samples/labelled_videos``.

.. _notebook-video-reader-with-label:

video_reader_with_label
=======================

This notebook is similar to ``video_decoder``, but it also returns the label of each video sequence.

* **Operators:** ``fn.readers.video`` reads and decodes the video as sequences of frames, and ``fn.crop_mirror_normalize`` processes the sequences.
* **Iterator:** The notebook defines a custom ``ROCALVideoIterator`` class that returns the labels together with the video sequences. The notebook plots the sequences using ``matplotlib``.
* **Data:** Set ``ROCAL_DATA_PATH``. The notebook reads the videos from ``${ROCAL_DATA_PATH}/rocal_data/video_and_sequence_samples/labelled_videos``.

.. _notebook-webdataset-reader:

webdataset_reader_example
=========================

This notebook shows how to set up a pipeline that loads and decodes image data that is stored in the WebDataset format. 
It resizes and normalizes the decoded images and displays them.

* **Operators:** ``fn.readers.webdataset`` reads the data. ``fn.decoders.image`` decodes the images. The notebook then applies ``fn.resize`` and ``fn.crop_mirror_normalize``.
* **Reader arguments:** ``fn.readers.webdataset`` takes the following arguments:

  * ``path``: The path to the WebDataset ``.tar`` files.
  * ``index_paths``: The path to the index files that contain information about the ``.tar`` files. If you don't provide a path, rocAL infers the information from the ``.tar`` files. To create index files, use the ``tar2idx`` tool in the `tools folder <https://github.com/ROCm/rocAL/tree/develop/tools>`_ of the rocAL repository.
  * ``ext``: The extensions of the types of data in the ``.tar`` files, passed as a list of sets of extensions. The notebook uses ``ext=[{'JPEG', 'cls'}]`` to read the image and its class label from each sample.
  * ``missing_components_behavior``: How the reader behaves when a sample in the dataset lacks a file that the reader expects to find. The notebook uses ``types.MISSING_COMPONENT_SKIP``.

* **Iterator:** ``ROCALClassificationIterator``, from ``amd.rocal.plugin.generic``.
* **Data:** Set ``ROCAL_DATA_PATH``. Create a ``web_dataset`` folder in the ``rocal_data`` folder. Put the ``.tar`` files in a subfolder named ``tar_file`` and the ``.idx`` index files in a subfolder named ``idx_file``. The notebook reads the data from ``${ROCAL_DATA_PATH}/rocal_data/web_dataset/tar_file/``.
