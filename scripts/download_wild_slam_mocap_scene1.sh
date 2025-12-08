#!/bin/bash

mkdir -p datasets/Wild_SLAM_Mocap/scene1
cd datasets/Wild_SLAM_Mocap/scene1

scenes=(
    "ball"
    "crowd"
    "person_tracking"
    "racket"
    "stones"
    "table_tracking1"
    "table_tracking2"
    "umbrella"
)

for scene in "${scenes[@]}"
do
    echo "Processing scene: $scene"
    
    # Check if the folder already exists
    if [ -d "$scene" ]; then
        echo "Folder $scene already exists, skipping download"
    else
        zip_file="${scene}.zip"
        wget "https://huggingface.co/datasets/gradient-spaces/Wild-SLAM/resolve/main/Mocap/scene1/${zip_file}"
        
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded ${zip_file}"
            unzip -q "${zip_file}"
            if [ $? -eq 0 ]; then
                echo "Successfully extracted ${zip_file}"
                rm "${zip_file}"
                echo "Removed ${zip_file}"
            else
                echo "Failed to extract ${zip_file}"
            fi
        else
            echo "Failed to download ${zip_file}"
        fi
    fi
    
    echo "Finished processing ${scene}"
    echo "-----------------------------"
done