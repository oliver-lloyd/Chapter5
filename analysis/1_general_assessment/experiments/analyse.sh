#!/bin/bash

echo Creating aggregates;
python ../../../analysis_scripts/get_centroid_aggregates.py

echo Getting cosines;
python ../../../analysis_scripts/get_cosine_with_actual.py;

echo Getting per-epoch runtime;
python ../../../analysis_scripts/get_per-epoch_runtime.py

mkdir -p holdout_scores
echo Getting polySE scores with aggregations;
python ../../../analysis_scripts/polySE_with_aggregates.py

echo Getting POSTHOC polySE scores with aggregations;
python ../../../analysis_scripts/polySE_with_aggregates.py --posthoc_removal

echo Getting polySE scores with actual embeds;
sh ../../../analysis_scripts/polySE_with_true_embeds.sh;