#!/usr/bin/env bash

# This is a script to run multiple job

# Flag to control the running mode
FLAG=2

if [[ "${FLAG}" == 1 ]]
then
  for (( i=1; i<11; i++ ))
  do
    D="${i}"
    python MFspark.py small_data 5 --N 40 --gain 0.001 --pow 0.2 --maxiter 20 --d $D 
  done
fi

if [[ "${FLAG}" == 2 ]]
then
  for (( i=0; i<=50; i=i+5 ))
    do
      LAM_MU="${i}"
      python MFspark.py small_data 5 --N 40 --gain 0.001 --pow 0.2 --maxiter 20 --d 4 --lam $LAM_MU --mu $LAM_MU
    done
fi
