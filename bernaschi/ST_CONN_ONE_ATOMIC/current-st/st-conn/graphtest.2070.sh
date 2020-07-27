#!/bin/bash

THREADS=128

for ((l=0;l<4;l++))
do
  for k in graphs/dense/*
  do
    echo "Analyzing" $(basename -- "$k")
    echo "THREADS: " $THREADS
    filename=$(basename -- "$k")
    extension="${filename##*.}"
    filename="${filename%.*}"
    ./st-con_mpi_cuda -f $k -t $THREADS | grep 'mean_time:' > times/dense/$filename.avg.2070.$THREADS.force_undirected.time.txt
	sleep 1
  done
  
  for k in graphs/streets/*
  do
    echo "Analyzing" $(basename -- "$k")
    echo "THREADS: " $THREADS
    filename=$(basename -- "$k")
    extension="${filename##*.}"
    filename="${filename%.*}"
    ./st-con_mpi_cuda -f $k -t $THREADS | grep 'mean_time:' > times/streets/$filename.avg.2070.$THREADS.force_undirected.time.txt
	sleep 1
  done
  
  for k in graphs/symmetric/*
  do
    echo "Analyzing" $(basename -- "$k")
    echo "THREADS: " $THREADS
    filename=$(basename -- "$k")
    extension="${filename##*.}"
    filename="${filename%.*}"
    ./st-con_mpi_cuda -f $k -t $THREADS | grep 'mean_time:' > times/symmetric/$filename.avg.2070.$THREADS.force_undirected.time.txt
	sleep 1
  done
  #aumento i threads per il prossimo ciclo
  THREADS=$((THREADS*2))
done