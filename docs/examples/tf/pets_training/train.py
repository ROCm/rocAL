# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from amd.rocal.plugin.tf import ROCALIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types


############################### CHANGE THESE GLOBAL VARIABLES APPROPRIATELY ###############################

RECORDS_DIR = 'tf_pets_records/'
NUM_CLASSES = 37
LEARNING_RATE = 0.005
TRAIN_BATCH_SIZE = 8
RUN_ON_HOST = True

############################### CHANGE THESE GLOBAL VARIABLES APPROPRIATELY ###############################


######################################## NO CHANGES IN CODE NEEDED ########################################

TRAIN_RECORDS_DIR = RECORDS_DIR + 'train/'
VAL_RECORDS_DIR = RECORDS_DIR + 'val/'

def main():

    global NUM_CLASSES
    global LEARNING_RATE
    global TRAIN_BATCH_SIZE
    global TRAIN_RECORDS_DIR
    global VAL_RECORDS_DIR

    print("\n-----------------------------------------------------------------------------------------")
    print('TF records (train) are located in %s' % TRAIN_RECORDS_DIR)
    print('TF records (val) are located in %s' % VAL_RECORDS_DIR)
    print("-----------------------------------------------------------------------------------------\n")

    image_size = [128, 128, 3]
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_size,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])

    model.summary()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['acc'])

    TFRecordReaderType = 0
    featureKeyMap = {
        'image/encoded': 'image/encoded',
        'image/class/label': 'image/object/class/label',
        'image/filename': 'image/filename'
    }

    trainPipe = Pipeline(batch_size=TRAIN_BATCH_SIZE, num_threads=8, rocal_cpu=RUN_ON_HOST,
                         tensor_layout=types.NHWC, mean=[0, 0, 0], std=[255, 255, 255], tensor_dtype=types.FLOAT)
    with trainPipe:
        inputs = fn.readers.tfrecord(path=TRAIN_RECORDS_DIR, reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
                                     features={
                                         'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                                         'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                                         'image/filename': tf.io.FixedLenFeature((), tf.string, "")
                                     }
                                     )
        jpegs = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        images = fn.decoders.image(
            jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=TRAIN_RECORDS_DIR)
        resized = fn.resize(
            images, resize_width=image_size[0], resize_height=image_size[1])
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmn_images = fn.crop_mirror_normalize(resized, crop=(image_size[1], image_size[0]),
                                              mean=[127.5, 127.5, 127.5],
                                              std=[127.5, 127.5, 127.5],
                                              mirror=flip_coin,
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NHWC)
        trainPipe.set_outputs(cmn_images)
    trainPipe.build()

    valPipe = Pipeline(batch_size=TRAIN_BATCH_SIZE, num_threads=8,
                       rocal_cpu=RUN_ON_HOST, tensor_layout=types.NHWC, tensor_dtype=types.FLOAT)
    with valPipe:
        inputs = fn.readers.tfrecord(path=VAL_RECORDS_DIR, reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap,
                                     features={
                                         'image/encoded': tf.io.FixedLenFeature((), tf.string, ""),
                                         'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                                         'image/filename': tf.io.FixedLenFeature((), tf.string, "")
                                     }
                                     )
        jpegs = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        images = fn.decoders.image(
            jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB, path=VAL_RECORDS_DIR)
        resized = fn.resize(
            images, resize_width=image_size[0], resize_height=image_size[1])
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmn_images = fn.crop_mirror_normalize(resized, crop=(image_size[1], image_size[0]),
                                              mean=[127.5, 127.5, 127.5],
                                              std=[127.5, 127.5, 127.5],
                                              mirror=flip_coin,
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NHWC)
        valPipe.set_outputs(cmn_images)
    valPipe.build()

    trainIterator = ROCALIterator(trainPipe)
    valIterator = ROCALIterator(valPipe)

    # Create the metrics
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_acc')
    epoch = 0
    train_batches = math.ceil(len(trainIterator) / TRAIN_BATCH_SIZE)
    val_batches = math.ceil(len(valIterator) / TRAIN_BATCH_SIZE)
    while epoch < 10:
        print('Epoch :', epoch + 1)
        accuracy_metric.reset_states()
        pbar = tf.keras.utils.Progbar(target=train_batches, stateful_metrics=[])
        step = 0
        for ([train_image_ndArray], train_label_ndArray) in trainIterator:
            train_label_ndArray = train_label_ndArray - 1
            with tf.GradientTape() as tape:
                prediction = model(train_image_ndArray, training=True)
                loss = loss_fn(train_label_ndArray, prediction)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))
                accuracy_metric.update_state(train_label_ndArray, prediction)
                results = {'loss': loss, 'train_acc': accuracy_metric.result()}
                step += 1
                pbar.update(step, values=results.items(), finalize=False)
        pbar.update(step, values=results.items(), finalize=True)
        trainIterator.reset()
        accuracy_metric.reset_states()
        pbar = tf.keras.utils.Progbar(target=val_batches, stateful_metrics=[])
        step = 0
        for ([val_image_ndArray], val_label_ndArray) in valIterator:
            val_label_ndArray = val_label_ndArray - 1
            prediction = model(val_image_ndArray, training=False)
            accuracy_metric.update_state(val_label_ndArray, prediction)
            results = {'val_acc': accuracy_metric.result()}
            step += 1
            pbar.update(step, values=results.items(), finalize=False)
        pbar.update(step, values=results.items(), finalize=True)
        valIterator.reset()
        epoch += 1


if __name__ == '__main__':
    main()

######################################## NO CHANGES IN CODE NEEDED ########################################
