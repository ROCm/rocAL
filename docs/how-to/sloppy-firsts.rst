.. meta::
  :description: Walkthrough of the rocAL PyTorch Toynet train.py example
  :keywords: rocAL, ROCm, API, PyTorch, training, machine learning, ML, walkthrough, example

.. _pytorch-train-walkthrough:

*******************************************************
Walkthrough of the PyTorch Toynet ``train.py`` example
*******************************************************

This page provides a complete walkthrough of |train.py|_, the Toynet image classification example that demonstrates how to build a GPU-accelerated training pipeline with rocAL and PyTorch. The script reads JPEG images organized into class sub-folders, applies a series of augmentations through a rocAL pipeline, and trains a small convolutional neural network (ToyNet) on the augmented data.


Imports
=======

.. code:: python

  import sys
  from amd.rocal.plugin.pytorch import ROCALClassificationIterator
  from amd.rocal.pipeline import Pipeline
  import amd.rocal.fn as fn
  import amd.rocal.types as types
  import os
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  import torch

The imports fall into two groups:

**rocAL imports:**

* ``ROCALClassificationIterator`` -- a PyTorch-compatible iterator that wraps a rocAL pipeline and yields ``(image_tensor, label_tensor)`` tuples each iteration. It handles running the pipeline, copying output tensors from rocAL's internal buffers into PyTorch tensors, and raising ``StopIteration`` when all images have been consumed.
* ``Pipeline`` -- the core rocAL class. It creates a processing context (CPU or GPU), manages the augmentation graph, and handles prefetch queuing.
* ``amd.rocal.fn`` -- contains the augmentation operators (readers, decoders, resize, crop-mirror-normalize, and so on). These are the building blocks of the augmentation graph.
* ``amd.rocal.types`` -- enumerations shared between rocAL's C++ backend and Python, including data types (``FLOAT``, ``FLOAT16``), tensor layouts (``NCHW``, ``NHWC``), color types (``RGB``), and process modes (``CPU``, ``GPU``).

**PyTorch imports:**

* ``torch.nn`` and ``torch.nn.functional`` -- used to define the ToyNet model architecture.
* ``torch.optim`` -- provides the SGD optimizer.
* ``torch`` -- general tensor operations and CUDA stream management.


``trainPipeline`` -- building the augmentation graph
=====================================================

.. code:: python

  def trainPipeline(data_path, batch_size, num_classes, one_hot, local_rank,
                    world_size, num_thread, crop, rocal_cpu, fp16):
      pipe = Pipeline(batch_size=batch_size, num_threads=num_thread,
                      device_id=local_rank, seed=local_rank+10,
                      rocal_cpu=rocal_cpu,
                      tensor_dtype=types.FLOAT16 if fp16 else types.FLOAT,
                      tensor_layout=types.NCHW,
                      prefetch_queue_depth=7)
      with pipe:
          jpegs, labels = fn.readers.file(file_root=data_path,
                                          shard_id=local_rank,
                                          num_shards=world_size,
                                          random_shuffle=True)
          decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,
                                           file_root=data_path,
                                           shard_id=local_rank,
                                           num_shards=world_size,
                                           random_shuffle=True)
          res = fn.resize(decode, resize_x=224, resize_y=224)
          flip_coin = fn.random.coin_flip(probability=0.5)
          cmnp = fn.crop_mirror_normalize(res, device="gpu",
                                          output_dtype=types.FLOAT,
                                          output_layout=types.NCHW,
                                          crop=(crop, crop),
                                          mirror=flip_coin,
                                          image_type=types.RGB,
                                          mean=[0.485*255, 0.456*255, 0.406*255],
                                          std=[0.229*255, 0.224*255, 0.225*255])
          if one_hot:
              _ = fn.one_hot(labels, num_classes)
          pipe.set_outputs(cmnp)
      return pipe

This is the heart of the data loading pipeline. It constructs a rocAL ``Pipeline`` object and populates its augmentation graph. Each step is explained below.

Creating the pipeline
---------------------

``Pipeline(...)`` creates a rocAL processing context. Key parameters:

* ``batch_size`` -- number of images per batch.
* ``num_threads`` -- CPU threads used for parallel decoding.
* ``device_id`` -- the GPU device index; in multi-GPU setups this is the local rank.
* ``seed`` -- random seed offset by ``local_rank`` so each GPU shard gets different random augmentations.
* ``rocal_cpu`` -- when ``True``, all operations run on the CPU; otherwise they run on the GPU.
* ``tensor_dtype`` -- output precision, ``FLOAT16`` for mixed-precision training or ``FLOAT`` for single-precision.
* ``tensor_layout`` -- ``NCHW`` (batch, channels, height, width), the layout PyTorch convolutions expect.
* ``prefetch_queue_depth`` -- how many batches rocAL decodes ahead of time. A depth of 7 keeps the GPU fed even when individual batches take variable time to decode and augment.

Reading files
-------------

``fn.readers.file`` reads JPEG file paths and their labels from a directory tree where each sub-directory name represents a class label. The ``shard_id`` / ``num_shards`` parameters let multiple workers split the dataset for distributed training. ``random_shuffle=True`` randomizes the reading order each epoch.

Decoding
--------

``fn.decoders.image_slice`` decodes the raw JPEG byte-streams into RGB image tensors. The ``image_slice`` variant supports partial (region-of-interest) decoding. This step converts compressed JPEG data into pixel arrays that subsequent operators can process.

Resizing
--------

``fn.resize(decode, resize_x=224, resize_y=224)`` resizes every decoded image to 224 x 224 pixels, the input size ToyNet expects.

Random horizontal flip
----------------------

``fn.random.coin_flip(probability=0.5)`` returns a random boolean for each image in the batch. This value is fed into the ``mirror`` parameter of the next operator so that roughly half of all images are horizontally flipped, a standard augmentation for image classification.

Crop, mirror, and normalize
----------------------------

``fn.crop_mirror_normalize`` performs three operations in a single fused GPU kernel:

1. **Crop** -- center-crops the image to ``(crop, crop)`` pixels. Because the images have already been resized to 224 x 224 and the crop size is also 224, this is effectively a no-op crop, but the operator is required for normalization.
2. **Mirror** -- conditionally flips the image horizontally based on ``flip_coin``.
3. **Normalize** -- subtracts the per-channel ImageNet mean (``[0.485, 0.456, 0.406]`` scaled to the ``[0, 255]`` range) and divides by the per-channel standard deviation (``[0.229, 0.224, 0.225]`` scaled likewise). The output is a float tensor in ``NCHW`` layout.

One-hot encoding (optional)
----------------------------

If ``one_hot`` is truthy, ``fn.one_hot(labels, num_classes)`` converts the integer class labels into one-hot vectors. This example leaves ``one_hot`` at ``0.0`` (disabled), so labels remain as integer indices, which is the format ``CrossEntropyLoss`` expects.

Setting outputs
---------------

``pipe.set_outputs(cmnp)`` tells the pipeline which tensor(s) to return to the caller. Only the augmented image tensor is returned here; labels are handled internally by the ``ROCALClassificationIterator``.


``trainLoader`` -- wrapping the pipeline for PyTorch
=====================================================

.. code:: python

  class trainLoader():
      def __init__(self, data_path, batch_size, num_thread, crop, rocal_cpu):
          super(trainLoader, self).__init__()
          self.data_path = data_path
          self.batch_size = batch_size
          self.num_thread = num_thread
          self.crop = crop
          self.rocal_cpu = rocal_cpu
          self.num_classes = 1000
          self.one_hot = 0.0
          self.local_rank = 0
          self.world_size = 1
          self.fp16 = True

      def get_pytorch_train_loader(self):
          print("in get_pytorch_train_loader function")
          pipe_train = trainPipeline(self.data_path, self.batch_size,
                                     self.num_classes, self.one_hot,
                                     self.local_rank, self.world_size,
                                     self.num_thread, self.crop,
                                     self.rocal_cpu, self.fp16)
          pipe_train.build()
          train_loader = ROCALClassificationIterator(
              pipe_train,
              device="cpu" if self.rocal_cpu else "cuda",
              device_id=self.local_rank)
          if self.rocal_cpu:
              return PrefetchedWrapper_rocal(train_loader, self.rocal_cpu), len(train_loader)
          else:
              return train_loader, len(train_loader)

``trainLoader`` is a convenience class that stores default hyper-parameters and exposes a single factory method, ``get_pytorch_train_loader``.

Default parameters
------------------

* ``num_classes = 1000`` -- matches the ImageNet class count. During actual training, the class count is detected dynamically in ``main()`` and passed to the model, but the pipeline uses this value only when ``one_hot`` encoding is enabled.
* ``local_rank = 0``, ``world_size = 1`` -- single-GPU defaults; for multi-GPU training these would come from the distributed launcher.
* ``fp16 = True`` -- the pipeline outputs half-precision tensors to reduce memory bandwidth and leverage GPU Tensor Cores.

``get_pytorch_train_loader``
----------------------------

1. Calls ``trainPipeline(...)`` to create the augmentation graph.
2. Calls ``pipe_train.build()`` -- this verifies the graph, allocates buffers, and makes the pipeline ready to run.
3. Wraps the built pipeline in a ``ROCALClassificationIterator``, which is a Python iterator that calls the rocAL C++ backend each step, copies output images and labels into PyTorch tensors, and raises ``StopIteration`` when all images have been consumed.
4. When running on CPU (``rocal_cpu=True``), the iterator is further wrapped in ``PrefetchedWrapper_rocal`` (explained next) to overlap CPU-to-GPU data transfer with computation. When running on GPU the iterator already produces CUDA tensors, so no extra wrapper is needed.
5. Returns the loader and its length (total batches per epoch).


``PrefetchedWrapper_rocal`` -- overlapping data transfer with computation
==========================================================================

.. code:: python

  class PrefetchedWrapper_rocal(object):
      def prefetched_loader(loader, rocal_cpu):
          stream = torch.cuda.Stream()
          first = True
          input = None
          target = None
          for next_input, next_target in loader:
              with torch.cuda.stream(stream):
                  if rocal_cpu:
                      next_input = next_input.cuda(non_blocking=True)
                      next_target = next_target.cuda(non_blocking=True)
              if not first:
                  yield input, target
              else:
                  first = False
              torch.cuda.current_stream().wait_stream(stream)
              input = next_input
              target = next_target
          yield input, target

      def __init__(self, dataloader, rocal_cpu):
          self.dataloader = dataloader
          self.epoch = 0
          self.rocal_cpu = rocal_cpu

      def reset(self):
          self.dataloader.reset()

      def __iter__(self):
          self.epoch += 1
          return PrefetchedWrapper_rocal.prefetched_loader(
              self.dataloader, self.rocal_cpu)

This class is only used when ``rocal_cpu=True`` (that is, when rocAL produces CPU tensors that need to be moved to the GPU for training).

``prefetched_loader`` (static generator)
-----------------------------------------

It creates a dedicated CUDA stream and uses it to asynchronously copy the *next* batch to the GPU while the *current* batch is being consumed by the training loop on the default CUDA stream. The ``non_blocking=True`` flag on ``.cuda()`` enables the asynchronous transfer. ``torch.cuda.current_stream().wait_stream(stream)`` ensures the transfer is complete before the training loop reads the tensor.

This is a classic double-buffering / pipeline prefetch pattern that hides the CPU-to-GPU transfer latency.

``reset``
---------

Delegates to the underlying ``ROCALClassificationIterator.reset()``, which calls ``rocalResetLoaders`` in the C++ backend to rewind the file reader to the beginning of the dataset.

``__iter__``
------------

Increments the epoch counter and returns the prefetching generator.


``ToyNet`` -- a small CNN for classification
==============================================

.. code:: python

  class ToyNet(nn.Module):
      def __init__(self, num_classes):
          super(ToyNet, self).__init__()
          self.conv1 = nn.Conv2d(3, 6, 5)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.conv3 = nn.Conv2d(16, 64, 3)
          self.conv4 = nn.Conv2d(64, 256, 3)
          self.fc0 = nn.Linear(256 * 11 * 11, 2048)
          self.fc1 = nn.Linear(2048, 512)
          self.fc2 = nn.Linear(512, 128)
          self.fc3 = nn.Linear(128, num_classes)
          self.m = nn.Softmax()

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = self.pool(F.relu(self.conv3(x)))
          x = self.pool(F.relu(self.conv4(x)))
          x = x.view(-1, 256 * 11 * 11)
          x = F.relu(self.fc0(x))
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

ToyNet is a minimal CNN meant for demonstration. It is not intended for production accuracy but is small enough to train quickly on any GPU.

Architecture summary (for a 224 x 224 x 3 input)
--------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Layer
     - Output shape
     - Description
   * - ``conv1`` + ReLU + pool
     - (batch, 6, 110, 110)
     - 5 x 5 conv (3 to 6 channels), then 2 x 2 max-pool
   * - ``conv2`` + ReLU + pool
     - (batch, 16, 53, 53)
     - 5 x 5 conv (6 to 16), then pool
   * - ``conv3`` + ReLU + pool
     - (batch, 64, 25, 25)
     - 3 x 3 conv (16 to 64), then pool
   * - ``conv4`` + ReLU + pool
     - (batch, 256, 11, 11)
     - 3 x 3 conv (64 to 256), then pool
   * - Flatten
     - (batch, 30976)
     - Reshape to 1-D
   * - ``fc0`` + ReLU
     - (batch, 2048)
     - Fully connected
   * - ``fc1`` + ReLU
     - (batch, 512)
     - Fully connected
   * - ``fc2`` + ReLU
     - (batch, 128)
     - Fully connected
   * - ``fc3``
     - (batch, num_classes)
     - Output logits (raw scores for each class)

Note that ``nn.Softmax`` is instantiated (``self.m``) but never called in ``forward()``. The softmax is unnecessary because ``nn.CrossEntropyLoss`` already applies log-softmax internally.


``main`` -- entry point and training loop
==========================================

.. code:: python

  def main():
      if len(sys.argv) < 4:
          print('Please pass image_folder cpu/gpu batch_size')
          exit(0)
      if sys.argv[2] == "cpu":
          rocal_cpu = True
      else:
          rocal_cpu = False
      bs = int(sys.argv[3])
      nt = 1
      crop_size = 224
      device = "cpu" if rocal_cpu else "cuda"
      image_path = sys.argv[1]
      dataset_train = image_path + '/train'
      num_classes = len(next(os.walk(image_path))[1])
      print("num_classes:: ", num_classes)

      net = ToyNet(num_classes)
      net.to(device)

      # Train loader
      train_loader_obj = trainLoader(dataset_train, batch_size=bs,
                                     num_thread=nt, crop=crop_size,
                                     rocal_cpu=rocal_cpu)
      train_loader, train_loader_len = train_loader_obj.get_pytorch_train_loader()

      criterion = nn.CrossEntropyLoss()
      optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

      # Training loop
      for epoch in range(10):
          print("\n epoch:: ", epoch)
          running_loss = 0.0

          for i, (inputs, labels) in enumerate(train_loader, 0):
              sys.stdout.write("\r Mini-batch " + str(i))
              inputs, labels = inputs.to(device), labels.to(device)
              optimizer.zero_grad()

              outputs = net(inputs)

              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

              running_loss += loss.item()
              print_interval = 10
              if i % print_interval == (print_interval - 1):
                  print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / print_interval))
                  running_loss = 0.0
          train_loader.reset()

      print('Finished Training')

Argument parsing
----------------

The script expects three positional command-line arguments:

1. ``image_folder`` -- root directory of the dataset. The images must be organized into class sub-folders (for example, ``image_folder/train/cat/``, ``image_folder/train/dog/``).
2. ``cpu`` or ``gpu`` -- selects whether rocAL processes data on the CPU or GPU.
3. ``batch_size`` -- number of images per mini-batch.

Class discovery
---------------

``num_classes = len(next(os.walk(image_path))[1])`` counts the number of sub-directories in the dataset root to automatically determine how many classes exist. This value configures the last fully connected layer of ToyNet.

Model instantiation
-------------------

``ToyNet(num_classes)`` is created and moved to the appropriate device (``cuda`` or ``cpu``).

Data loader creation
--------------------

``trainLoader`` is instantiated with the training data path and hyper-parameters, and ``get_pytorch_train_loader()`` builds and returns the ready-to-iterate rocAL pipeline.

Loss and optimizer
------------------

* ``nn.CrossEntropyLoss()`` combines log-softmax and negative log-likelihood loss in a single numerically stable operation.
* ``optim.SGD`` with a learning rate of ``0.0005`` and momentum of ``0.9`` is a conservative but stable optimizer suitable for demonstration.

The training loop (10 epochs)
------------------------------

1. The outer ``for epoch`` loop runs 10 times.
2. The inner ``for i, (inputs, labels)`` loop iterates over every mini-batch produced by the rocAL pipeline. Each iteration the ``ROCALClassificationIterator`` (or ``PrefetchedWrapper_rocal``) runs the next step of the pipeline and returns a batch of augmented images and their labels as PyTorch tensors.
3. Standard PyTorch training steps follow:

   a. ``optimizer.zero_grad()`` -- clears gradients from the previous step.
   b. ``net(inputs)`` -- forward pass through ToyNet.
   c. ``criterion(outputs, labels)`` -- computes the cross-entropy loss.
   d. ``loss.backward()`` -- computes gradients via back-propagation.
   e. ``optimizer.step()`` -- updates model weights.

4. The running loss is accumulated and printed every 10 mini-batches to monitor convergence.
5. At the end of each epoch, ``train_loader.reset()`` rewinds the rocAL file reader back to the beginning of the dataset so the next epoch sees all images again (in a new random order).


End-to-end data flow summary
==============================

The diagram below summarizes how data flows from disk to model::

  JPEG files on disk
       |
       v
  fn.readers.file          --  reads file paths and integer labels
       |
       v
  fn.decoders.image_slice  --  decodes JPEG bytes to RGB pixel tensors
       |
       v
  fn.resize                --  resizes to 224 x 224
       |
       v
  fn.crop_mirror_normalize --  center-crop, random mirror, ImageNet normalization
       |
       v
  ROCALClassificationIterator  --  copies rocAL tensors to PyTorch tensors
       |
       v
  ToyNet (forward pass -> loss -> backward -> optimizer step)


Running the example
====================

.. code:: bash

  python train.py <image_folder> <cpu/gpu> <batch_size>

For example, to train on a dataset stored in ``/data/imagenet`` using the GPU with a batch size of 32:

.. code:: bash

  python train.py /data/imagenet gpu 32

The dataset directory must contain a ``train/`` sub-folder, and the sub-folder must contain one directory per class, each filled with JPEG images. This is the standard ImageNet-style directory layout.


.. |train.py| replace:: ``train.py``
.. _train.py: https://github.com/ROCm/rocAL/tree/develop/docs/examples/pytorch/toynet_training/train.py
