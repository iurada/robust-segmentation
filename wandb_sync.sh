#!/bin/bash

. $WORK/activate.sh

while :
do
    for x in 1 2
    do
      echo -e "\nSyncing directory $x..\n"
      wandb sync training$x/wandb/$(ls -1 training$x/wandb | tail -n 1)
      echo -e "\n"
    done
    sleep 5m
done
