#!/usr/bin/env bash

# This is a script to run multiple job

for (( i=0; i<1; i++ ))
do
  LAM="${i}.0"
  python ParallelRegression.py --train data/small.train --test data/small.test --beta beta_small_0.0 --lam 0.0 --N 4 --silent
done
