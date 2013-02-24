#!/bin/bash

# Extra-trees
n_estimators=500

for max_features in 2500 3000 3500 3725
do
for min_samples_split in 1 5 10 25 50
do

echo qsub -o grid/extratrees -e grid/extratrees grid-job.sh extratrees $n_estimators $max_features $min_samples_split

done
done

# Random forest
n_estimators=500

for max_features in 1 500 1000 2000 3000 3725
do
for min_samples_split in 1 10 20 30
do

echo qsub -o grid/randomforest -e grid/randomforest grid-job.sh randomforest $n_estimators $max_features $min_samples_split

done
done

# GBRT
n_estimators=500

for max_depth in 3 5
do
for max_features in 3725 256 64
do
for learning_rate in 0.1 0.02
do
for min_samples_split in 7 13
do

echo qsub -o grid/gbrt -e grid/gbrt grid-job.sh gbrt $n_estimators $max_depth $learning_rate $max_features $min_samples_split

done
done
done
done

