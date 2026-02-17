# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import random
import sys
import os
import cv2
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

def save_output_images(img, idx, output_dir, device=True, layout="NCHW"):
    """Save output images for verification"""
    if device is False:
        image = img.cpu().numpy()
    else:
        image = img.numpy()
    if layout == "NCHW":
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"serialize_test_{idx}.png"), image)


def create_test_pipeline(data_path, rocal_cpu=True, batch_size=2):
    """Create a test pipeline with supported augmentations"""
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    local_rank = 0
    world_size = 1

    # Create pipeline
    pipeline = Pipeline(
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=device_id, 
        seed=random_seed, 
        rocal_cpu=rocal_cpu
    )

    with pipeline:
        # File reader
        jpegs, labels = fn.readers.file(file_root=data_path)
        
        # Image decoder
        decode = fn.decoders.image(
            jpegs, 
            output_type=types.RGB,
            file_root=data_path, 
            shard_id=local_rank, 
            num_shards=world_size, 
            random_shuffle=False
        )
        
        # Brightness augmentation
        brightness = fn.brightness(
            decode,
            brightness=0.5,
            output_layout=types.NCHW,
            output_dtype=types.UINT8
        )
        
        pipeline.set_outputs(brightness)

    return pipeline


def test_serialization(data_path, rocal_cpu=True, batch_size=2):
    """Test pipeline serialization functionality and return serialized string"""
    print(f">>> Testing Pipeline Serialization on {'CPU' if rocal_cpu else 'GPU'}")
    
    # Create output directory
    output_dir = "output_folder/serialize_test"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as error:
        print(f"Error creating output directory: {error}")
        return None

    try:
        # Create and build pipeline
        print("Creating test pipeline...")
        pipeline = create_test_pipeline(data_path, rocal_cpu, batch_size)
        pipeline.build()

        # Test serialization
        print("\n=== Testing Pipeline Serialization ===")
        
        # Test Serialize to string
        print("Test 1: Serializing pipeline to string...")
        serialized_string = pipeline.serialize()
        
        if serialized_string is None or len(serialized_string) == 0:
            print("ERROR: Failed to serialize pipeline - empty result")
            return None
            
        print(f"Serialized string size: {len(serialized_string)} bytes")
        print("Serialization to string: SUCCESS")
        
        # Display serialized content (first 500 chars for readability)
        print("\n=== Serialized Pipeline Content (Preview) ===")
        try:
            # Try to decode as text for preview
            preview_text = serialized_string.decode('utf-8', errors='ignore')[:500]
            print(preview_text)
            if len(serialized_string) > 500:
                print("... (truncated)")
        except Exception:
            # If binary, show hex representation
            print("Binary content (hex preview):")
            print(serialized_string[:100].hex())
            if len(serialized_string) > 100:
                print("... (truncated)")
        print("=== End of Serialized Content Preview ===")
        
        # Test pipeline execution after serialization
        print("\n=== Testing Pipeline Execution After Serialization ===")
        
        imageIteratorPipeline = ROCALClassificationIterator(pipeline)
        print(f"Available images: {pipeline.get_remaining_images()}")
        
        iteration_count = 0        
        for i, batch_data in enumerate(imageIteratorPipeline):
                
            print(f"\nIteration {iteration_count + 1}:")
            images, labels = batch_data
            
            print(f"  Batch shape: {images[0].shape} images")
            print(f"  Labels: {labels}")
            
            # Save output images
            if len(images) > 0:
                save_output_images(
                    images[0][0],
                    f"serialization_{iteration_count}", 
                    output_dir, 
                    device=rocal_cpu, 
                    layout="NCHW"
                )
                print(f"  Saved output image: serialization_{iteration_count}.png")
                
            iteration_count += 1
            
        imageIteratorPipeline.reset()
        print("\n=== Serialization Test Completed Successfully ===")
        return serialized_string
        
    except Exception as e:
        print(f"ERROR: Exception during serialization test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_deserialization(serialized_string, rocal_cpu=True, batch_size=2):
    """Test pipeline deserialization functionality"""
    print(f">>> Testing Pipeline Deserialization on {'CPU' if rocal_cpu else 'GPU'}")
    
    # Create output directory
    output_dir = "output_folder/serialize_test"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as error:
        print(f"Error creating output directory: {error}")
        return False

    try:
        if serialized_string is None:
            print("ERROR: No serialized string provided")
            return False
            
        print(f"Received serialized string of size: {len(serialized_string)} bytes")
        
        # Test deserialization from string
        print("\n=== Testing Pipeline Deserialization from String ===")
        
        try:
            deserialized_pipeline = Pipeline.deserialize(serialized_pipeline=serialized_string)
            print("Deserialization from string: SUCCESS")
        except Exception as e:
            print(f"ERROR: Failed to deserialize from string: {str(e)}")
            return False
        
        # Run deserialized pipeline from string and dump outputs
        print("\n=== Running Deserialized Pipeline (String) and Dumping Outputs ===")
        
        imageIteratorDeserialized = ROCALClassificationIterator(deserialized_pipeline)
        print(f"Available images: {deserialized_pipeline.get_remaining_images()}")
        
        iteration_count = 0
        
        for i, batch_data in enumerate(imageIteratorDeserialized):
            print(f"\nDeserialized (String) - Iteration {iteration_count + 1}:")
            images, labels = batch_data
            
            print(f"  Batch shape: {images[0].shape} images")
            print(f"  Labels: {labels}")
            
            # Save output images
            if len(images) > 0:
                save_output_images(
                    images[0][0],
                    f"deserialization_{iteration_count}", 
                    output_dir, 
                    device=rocal_cpu, 
                    layout="NCHW"
                )
                print(f"  Saved output image: deserialization_{iteration_count}.png")
                
            iteration_count += 1
            
        imageIteratorDeserialized.reset()
        print("\n=== Deserialization Test Completed Successfully ===")
        return True

    except Exception as e:
        print(f"ERROR: Exception during deserialization test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run serialization and deserialization tests"""
    if len(sys.argv) < 2:
        print('Usage: python pipeline_serialize_test.py <image_folder> [cpu/gpu] [batch_size]')
        sys.exit(1)
    
    # Parse arguments
    data_path = sys.argv[1]
    rocal_cpu = (sys.argv[2].lower() == "cpu") if len(sys.argv) > 2 else True
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    # Validate data path
    if not os.path.exists(data_path):
        print(f"ERROR: Data path does not exist: {data_path}")
        sys.exit(1)
    
    # Run the serialization test
    serialized_string = test_serialization(data_path, rocal_cpu, batch_size)

    if (serialized_string is None) or (len(serialized_string) == 0):
        print("SERIALIZATION TEST FAILED - No valid serialized string produced")
        sys.exit(1)

    # Run the deserialization test
    success = test_deserialization(serialized_string, rocal_cpu, batch_size)
    
    if success:
        print("SERIALIZATION AND DESERIALIZATION TESTS PASSED!")
    else:
        print("SERIALIZATION OR DESERIALIZATION TESTS FAILED")


if __name__ == '__main__':
    main()
