#!/bin/bash

for dir in checkpoint_*; 
do 
    cd $dir
    python ../../../../scripts/get_centroid_aggregates.py
    cd -
done