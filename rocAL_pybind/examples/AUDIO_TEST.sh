#!/bin/bash

# Function to compare binary files
compare_binary_files() {
    file1="$1"
    file2="$2"

    # Compare the contents of the two files
    if cmp -s "$file1" "$file2"; then
        echo "Files $file1 and $file2 are identical."
    else
        echo "Files $file1 and $file2 are different."
    fi
}

# Function to compare contents of two folders
compare_folders() {
    folder1="$1"
    folder2="$2"

    # Iterate over the files in folder1
    for file1 in "$folder1"/*; do
        # Construct the corresponding file path in folder2
        file2="$folder2/$(basename "$file1")"
        
        # Check if the file exists in folder2
        if [ -e "$file2" ]; then
            # Compare the files
            compare_binary_files "$file1" "$file2"
        else
            echo "File $(basename "$file1") does not exist in $folder2."
        fi
    done
}

# TODO: To change this a generic path from MIVsison-data
# Paths to the two folders
folder1="/media/swetha/audio_new_repo/rpp/utilities/test_suite/REFERENCE_OUTPUTS_AUDIO/pre_emphasis_filter"
folder2="OUTPUTS_AUDIO/BIN_OUTPUTS/pre_emphasis_filter"

# Compare the contents of the two folders
compare_folders "$folder1" "$folder2"
