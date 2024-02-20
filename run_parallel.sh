#!/bin/bash

DATASET="ExtraSensory"

NCLIENTS=60

NROUNDS=100

NCLUSTERS="1 5 10 30"

CLUSTERING="True"

CLUSTERROUND=10

NONIID="True"

SELECTIONMETHOD="All Random Less_Selected POC"

CLUSTERMETRIC="CKA weights"

METRICLAYER="-1"

POCPERCOFCLIENTS=0.5

CLUSTERMETHOD="HC Random KCenter"

parallel  -j5 --bar "python3 simulation.py --dataset {1} --nclients {2} \
 --nrounds {3} --nclusters {4} --clustering {5} --clusterround {6} --noniid {7} \
 --selectionmethod {8} --clustermetric {9} --metriclayer {10} --pocpercofclients {11} \
 --clustermethod {12} " ::: $DATASET ::: $NCLIENTS ::: $NROUNDS ::: \
 $NCLUSTERS ::: $CLUSTERING ::: $CLUSTERROUND ::: $NONIID :::\
 $SELECTIONMETHOD ::: $CLUSTERMETRIC ::: $METRICLAYER :::\
 $POCPERCOFCLIENTS ::: $CLUSTERMETHOD