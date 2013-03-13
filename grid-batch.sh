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

for n_features in 2000 
do
for max_depth in 8 
do
for max_features in 0.01 0.05 0.1
do
for learning_rate in  0.13  
do
for min_samples_split in 28 
do
for subsample in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do

qsub -o grid/gbrt -e grid/gbrt grid-job.sh $n_features gbrt $n_estimators $max_depth $learning_rate $max_features $min_samples_split $subsample

done
done
done
done
done
done

# DBN

for clip in 1000
do
for epochs in 50
do
for units in 300 400 500
do
for learn_rates in  0.008 0.007 0.006 0.005 0.004
do
for momentum in 0.1 0.08 0.06 0.04
do
for nc in 10
do

echo qsub -o grid/dbn -e grid/dbn grid-job.sh $clip $nc dbn $units-$units-$units $epochs $learn_rates $momentum

done
done
done
done
done
done

# Multiframe

for nftt in 1024 2048
do
for noverlap in 0 32 64 128 256 512 768
do
for clip in 100 250 500 1000
do

echo qsub -o grid/multi -e grid/multi grid-job.sh $nftt $noverlap $clip

done
done
done
