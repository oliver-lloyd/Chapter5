#!/bin/bash

experiment=`pwd`
cd ../../../Chapter3/analysis/assessment

python assessment.py $experiment/checkpoint_best.pt $experiment --partial_results results_temp.csv
cd $experiment
mv results_full.csv polySE_results_real_embeds.csv
