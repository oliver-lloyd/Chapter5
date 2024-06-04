#!/bin/bash

for f in checkpoint_*; 
do 
    f_dir=${f:0:-3}
    mkdir $f_dir;
    mv $f $f_dir 
done