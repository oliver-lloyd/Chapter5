#!/bin/bash



for dir in checkpoint_*; 
do 
    cd $dir
    if ! ls | grep holdoutscores;
    then
        mkdir holdout_scores;
    fi;
    python ../../../../scripts/polySE_with_aggregates.py;
    cd -;
done