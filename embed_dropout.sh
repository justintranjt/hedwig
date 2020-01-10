#!/usr/bin/env bash

dataset="$1"

if [[ -z "$dataset" ]]; then
  echo "Usage: $0 <dataset>"
  echo "Available datasets:"
  echo " - Reuters"
  echo " - AAPD"
  echo " - IMDB"
  echo " - Yelp2014"
  exit
fi

LR=0.001 # for all but reuters
BATCH=32 # multi label, 64 for single

if [[ "$dataset" == "Reuters" ]]; then
  # Multilabel, 90 classes
  LR=0.01

elif [[ "$dataset" == "AAPD" ]]; then
  # Multilabel, 54 classes
  true
elif [[ "$dataset" == "IMDB" ]]; then
  # Single label, 10 classes
  BATCH=64
elif [[ "$dataset" == "Yelp2014" ]]; then
  # Single label, 5 classes
  BATCH=64
else
  echo "Unknown dataset! Aborting..."
  exit
fi

echo "Training on the $dataset dataset, using LR=$LR, BatchSize=$BATCH..."

# Create logging dir if not present
if [[ ! -d "logs" ]]; then
    mkdir logs
fi

# Loop over various embedding dropouts
for i in $(seq 0.1 0.1 0.5)
do
  echo "Starting a new training run with a dropout of $i"
  ./train.sh $dataset $i
done
 
