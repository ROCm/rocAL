import types
import numpy as np
from random import shuffle
from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import cv2
import os


def main():

    batch_size = 5
    data_dir = os.environ["ROCAL_DATA_PATH"] + \
        "/coco/coco_10_img/train_10images_2017/"
    device = "cpu"
    try:
        path = "OUTPUT_IMAGES_PYTHON/EXTERNAL_SOURCE_READER/MODE2/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    def draw_patches(image, idx, device="cpu"):
        # image is expected as a tensor, bboxes as numpy
        import cv2
        if device == "gpu":
            image = image.cpu().detach().numpy()
        else:
            image = image.detach().numpy()
        image = image.transpose([1, 2, 0])  # NCHW
        image = (image).astype('uint8')
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/EXTERNAL_SOURCE_READER/MODE2/" +
                    str(idx)+"_"+"train"+".png", image)

    # Define the Data Source for all image samples

    class ExternalInputIteratorMode2(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            self.maxHeight = self.maxWidth = 0
            import os
            import glob
            for filename in glob.glob(os.path.join(self.images_dir, '*.jpg')):
                self.files.append(filename)
            shuffle(self.files)
            self.i = 0
            self.n = len(self.files)
            for x in range(self.n):
                jpeg_filename = self.files[x]
                label = 1
                image = cv2.imread(jpeg_filename, cv2.IMREAD_COLOR)
                # Check if the image was loaded successfully
                if image is None:
                    print("Error: Failed to load the image.")
                else:
                    # Get the height and width of the image
                    height, width = image.shape[:2]
                self.maxHeight = height if height > self.maxHeight else self.maxHeight
                self.maxWidth = width if width > self.maxWidth else self.maxWidth

        def __iter__(self):
            return self

        def __next__(self):
            batch = []
            batch_of_numpy = []
            labels = []
            label = 1
            roi_height = []
            roi_width = []
            self.out_image = np.zeros(
                (self.batch_size, self.maxHeight, self.maxWidth, 3), dtype="uint8")
            for x in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                image = cv2.imread(jpeg_filename, cv2.IMREAD_COLOR)
                # Check if the image was loaded successfully
                if image is None:
                    print("Error: Failed to load the image.")
                else:
                    # Get the height and width of the image
                    height, width = image.shape[:2]
                batch.append(np.asarray(image))
                roi_height.append(height)
                roi_width.append(width)
                self.out_image[x][:roi_height[x], :roi_width[x], :] = batch[x]
                batch_of_numpy.append(self.out_image[x])
                labels.append(label)
                label = label + 1
                self.i = (self.i + 1) % self.n
            labels = np.array(labels).astype('int32')
            return (batch_of_numpy, labels, roi_height, roi_width, self.maxHeight, self.maxWidth)


# Mode 2
    eii_2 = ExternalInputIteratorMode2(batch_size)

    # Create the pipeline
    external_source_pipeline_mode2 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0,
                                              seed=1, rocal_cpu=True if device == "cpu" else False, tensor_layout=types.NCHW)

    with external_source_pipeline_mode2:
        jpegs, _ = fn.external_source(source=eii_2, mode=types.EXTSOURCE_RAW_UNCOMPRESSED,
                                      max_width=eii_2.maxWidth, max_height=eii_2.maxHeight)
        output = fn.resize(jpegs, resize_width=300, resize_height=300,
                           output_layout=types.NCHW, output_dtype=types.UINT8)
        external_source_pipeline_mode2.set_outputs(output)

    # build the external_source_pipeline_mode2
    external_source_pipeline_mode2.build()
    # Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(
        external_source_pipeline_mode2, device=device)
    for i, output_list in enumerate(data_loader, 0):
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nImages:\n", output_list)
        print("**************ends*******************")
        print("**************", i, "*******************")
        for img in output_list[0][0]:
            cnt = cnt+1
            draw_patches(img, cnt, device=device)


if __name__ == '__main__':
    main()
