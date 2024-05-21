#!/bin/bash

base_dir_name=$1
mkdir $base_dir_name
for n in 5 10 20;
do
    dir_name="${base_dir_name}_n$n"
    mkdir $base_dir_name/$dir_name;
    for config in `ls ../config_templates`;
    do
        cp ../config_templates/$config $base_dir_name/$dir_name/"${dir_name}_${config}";
        sed -i "s/replace_me/${base_dir_name}_nearest${n}.pt/g" $base_dir_name/$dir_name/"${dir_name}_${config}";
    done
done